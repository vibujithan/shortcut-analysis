import gc
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.auto import tqdm

from models.macaw import MACAW

import sys

sys.path.append("/data/Code/bias-analysis/")
from utils.datasets import TorchDataset as TD


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
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

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

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

            # Zero gradients
            self.optimizer.zero_grad()

            with autocast():
                zs, prior_log_prob, log_det = self.model(batch.to(self.device))
                loss = -torch.mean(prior_log_prob + log_det)

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

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            with autocast():
                z, prior_log_prob, log_det = self.model(batch.to(self.device))
                loss = -torch.mean(prior_log_prob + log_det)

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
            writer = SummaryWriter(f"PD/runs/{self.save_dir}")

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
                writer.add_scalars("losses", metrics, epoch)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)


def main():
    block = "flat"
    model_name = "PD-SFCN-GroupDRO"
    train_path = f"/data/Data/PD/{model_name}_activations_{block}_train"
    val_path = f"/data/Data/PD/{model_name}_activations_{block}_val"

    batch_size = 64
    latent_size = 64

    train_loader = DataLoader(TD(train_path), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TD(val_path), batch_size=batch_size)

    model = MACAW(latent_size)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        weight_decay=1e-5,
        gradient_clip_val=1.0,
        save_dir=f"PD/checkpoints/{model_name}-act-macaw-{block}",
        device="cuda",
        use_tb=True,
    )

    # Train model
    trainer.train(
        num_epochs=3000,
    )


if __name__ == "__main__":
    main()
