import sys

sys.path.append("/data/Code/bias-analysis/")

import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from PD.models.sfcn import SFCN
from utils.datasets import TorchDataset as TD
import numpy as np


class LossComputer:
    def __init__(
        self,
        criterion,
        is_robust,
        n_groups,
        group_counts,
        alpha,
        gamma,
        adj=None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
    ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = n_groups
        self.group_counts = group_counts
        self.group_counts = self.group_counts.cuda()
        self.group_frac = self.group_counts / self.group_counts.sum()
        # self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, "alpha must be specified"

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        # group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        group_acc, group_count = self.compute_group_avg((yhat > 0.5).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (
            group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()
        ).float()  # size: 2 x batch_size
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        # import pdb; pdb.set_trace()

        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def update_stats(
        self, actual_loss, group_loss, group_acc, group_count, weights=None
    ):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = (
            prev_weight * self.avg_group_loss + curr_weight * group_loss
        )

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
            1 / denom
        ) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_computer,
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_clip_val=1.0,
        save_dir="checkpoints",
        device="cuda",
        use_tb=True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_tb = use_tb

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.train_loss_computer = loss_computer

        # Mixed precision training
        self.scaler = GradScaler()
        self.gradient_clip_val = gradient_clip_val

        # Checkpointing
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        self.best_val_loss = float("inf")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pt")

        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint["epoch"], checkpoint["val_loss"]

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        batch_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            torch.cuda.empty_cache()
            gc.collect()
            # Move batch to device
            x = batch[0].to(self.device)
            y = batch[1].float().to(self.device)
            y = torch.squeeze(y)
            sensitive_attr = batch[3].to(self.device)
            sensitive_attr = torch.squeeze(sensitive_attr)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                y_pred = self.model(x).squeeze()
                loss = self.train_loss_computer.loss(
                    y_pred, y, sensitive_attr, is_training=True
                )

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )

            # Optimizer step with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / batch_count

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            x = batch[0].to(self.device)
            y = batch[1].float().to(self.device)
            y = torch.squeeze(y)

            sensitive_attr = batch[3].to(self.device)
            sensitive_attr = torch.squeeze(sensitive_attr)

            # Forward pass with mixed precision
            with autocast():
                y_pred = self.model(x).squeeze()
                loss = self.train_loss_computer.loss(
                    y_pred, y, sensitive_attr, is_training=False
                )

            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count

    def train(self, num_epochs, resume_from=None):
        start_epoch = 0

        # Resume training if checkpoint provided
        if resume_from is not None:
            start_epoch, self.best_val_loss = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch}")

        if self.use_tb:
            writer = SummaryWriter("runs")

        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_epoch(epoch)

            val_loss = self.validate()

            self.scheduler.step(val_loss)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            if self.use_tb:
                writer.add_scalars("recon_losses", metrics, epoch)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)


def main():
    train_df = pd.read_csv("/data/Data/PD/train.csv")
    sens_name = "Site"

    n_groups = train_df[sens_name].nunique()
    group_counts = train_df[sens_name].value_counts().sort_index().values

    print(n_groups, group_counts)

    train_path = "/data/Data/PD/train"
    val_path = "/data/Data/PD/val"

    batch_size = 8

    train_loader = DataLoader(TD(train_path), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TD(val_path), batch_size=batch_size)

    model = SFCN(output_dim=1, channel_number=[28, 58, 128, 256, 256, 64])

    generalization_adjustment = "2"
    adjustments = [float(c) for c in generalization_adjustment.split(",")]
    assert len(adjustments) in (1, 9)  # 9 is hard-coded for Site
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * 9)
    else:
        adjustments = np.array(adjustments)

    loss_computer = LossComputer(
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        is_robust=True,
        alpha=0.1,
        gamma=1,
        n_groups=9,
        group_counts=torch.tensor([61, 122, 57, 36, 21, 22, 32, 18, 47]),
        adj=adjustments,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
        min_var_weight=0,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_computer=loss_computer,
        learning_rate=1e-4,
        weight_decay=1e-5,
        gradient_clip_val=1.0,
        save_dir="PD/checkpoints/PD-SFCN-GroupDRO",
        device="cuda",
        use_tb=True,
    )

    # Train model
    trainer.train(
        num_epochs=500,
        resume_from="PD/checkpoints/PD-SFCN-GroupDRO/latest_checkpoint.pt",
    )


if __name__ == "__main__":
    main()
