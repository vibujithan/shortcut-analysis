import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import sys

sys.path.append("/data/Code/bias-analysis/")
from PD.models.sfcn import SFCN

from utils.datasets import TorchDataset as TD


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        label_idx,
        class_idx,
        learning_rate=1e-4,
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
        self.label_idx = label_idx
        self.class_idx = class_idx

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision training
        self.scaler = GradScaler()
        self.gradient_clip_val = gradient_clip_val

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

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            torch.cuda.empty_cache()
            gc.collect()
            # Move batch to device
            x = batch[0].to(self.device)
            y = 1 * (batch[self.label_idx] == self.class_idx).to(self.device)
            y = torch.unsqueeze(y, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y.to(torch.float))

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
            y = 1 * (batch[self.label_idx] == self.class_idx).to(self.device)
            y = torch.unsqueeze(y, 1)

            # Forward pass with mixed precision
            with autocast():
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y.to(torch.float))

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
            writer = SummaryWriter("PD/runs")

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
    model_name = "PD-SFCN-GroupDRO"
    train_path = "/data/Data/PD/train"
    val_path = "/data/Data/PD/val"

    batch_size = 8
    label_idx = 3

    train_loader = DataLoader(TD(train_path), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TD(val_path), batch_size=batch_size)

    classes = np.arange(9)

    for c in classes:
        model = SFCN(output_dim=1, channel_number=[28, 58, 128, 256, 256, 64])
        model.load_state_dict(
            torch.load(f"PD/checkpoints/{model_name}/best_model.pt")["model_state_dict"]
        )

        for name, param in model.named_parameters():
            param.requires_grad = False

        model.classifier.fc.weight.requires_grad = True
        model.classifier.fc.bias.requires_grad = True

        save_dir = f"PD/checkpoints/{model_name}-study-{label_idx}-{c}"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_idx=label_idx,
            class_idx=c,
            learning_rate=1e-4,
            weight_decay=1e-5,
            gradient_clip_val=1.0,
            save_dir=save_dir,
            device="cuda",
            use_tb=True,
        )

        checkpoint_path = f"{save_dir}/latest_checkpoint.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = None

        # Train model
        trainer.train(
            num_epochs=100,
            resume_from=checkpoint_path,
        )


if __name__ == "__main__":
    main()
