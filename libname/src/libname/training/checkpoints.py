from pathlib import Path
import torch


def create_checkpoint_dir(path):
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(path, model, optimizer, epoch):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, path)


def load_checkpoint(path, model, optimizer, device, weights_only=True):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=weights_only
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


def save_best_checkpoint(
    path,
    model,
    optimizer,
    val_loss,
    best_val_loss
):
    val_loss = float(val_loss)

    if val_loss < best_val_loss:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss
        }, path)

        best_val_loss = val_loss

    return best_val_loss