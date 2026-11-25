import torch
import torch.nn as nn

from .checkpoints import (
    create_checkpoint_dir,
    save_checkpoint,
    save_best_checkpoint
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(
        self,
        model,
        num_epochs,
        optimizer,
        loss,
        checkpoint=False,
        best_checkpoint=False,
        device=device
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = loss
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.best_checkpoint = best_checkpoint
        self.device = device

        self.model.to(self.device)

    def fit(self, train_loader, val_loader=None):

        best_val_loss = float("inf")

        # Create checkpoint directories only when needed
        if self.checkpoint:
            checkpoint_dir = create_checkpoint_dir("checkpoints/epochs")

        if self.best_checkpoint:
            if val_loader is None:
                raise ValueError(
                    "A validation loader is required for best checkpoint saving."
                )

            best_dir = create_checkpoint_dir("checkpoints/best")
            best_path = best_dir / "best_checkpoint.pt"

        for epoch in range(self.num_epochs):

            self.model.train()

            if epoch == 0:
                print("Training Started...\n")

            total_loss = 0.0
            num_batches = 0

            # -------- TRAINING --------
            for x, y in train_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                B, T, V = logits.shape

                logits = logits.contiguous().view(B * T, V)
                y = y.contiguous().view(B * T)

                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if num_batches == 0:
                raise ValueError("The training loader is empty.")

            train_loss = total_loss / num_batches

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} "
                f"| Train Loss: {train_loss:.4f}",
                end=""
            )

            # -------- REGULAR CHECKPOINT --------
            if self.checkpoint:

                checkpoint_path = (
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                )

                save_checkpoint(
                    path=checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1
                )

            # -------- VALIDATION --------
            if val_loader is not None:

                val_loss = self.evaluate(val_loader)

                print(f" | Val Loss: {val_loss:.4f}", end="")

                # -------- BEST CHECKPOINT --------
                if self.best_checkpoint:

                    best_val_loss = save_best_checkpoint(
                        path=best_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        val_loss=val_loss,
                        best_val_loss=best_val_loss
                    )

            print()

    def evaluate(self, val_loader):

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                B, T, V = logits.shape

                logits = logits.contiguous().view(B * T, V)
                y = y.contiguous().view(B * T)

                loss = self.criterion(logits, y)

                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError("The validation loader is empty.")

        avg_loss = total_loss / num_batches

        return avg_loss