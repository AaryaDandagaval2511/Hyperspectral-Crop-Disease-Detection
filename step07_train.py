"""
STEP 7 — Training
==================
Full training loop with:
  • CrossEntropyLoss
  • Adam optimiser with cosine-annealing LR schedule
  • Per-epoch train loss / accuracy and validation loss / accuracy
  • Early stopping (patience = 15 epochs)
  • Best-model checkpoint saved to  outputs/checkpoints/best_model.pt

All metrics are logged to  outputs/training_log.csv  and the loss /
accuracy curves are saved to  outputs/visualizations/training_curves.png.
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "outputs/checkpoints"
VIZ_DIR        = "outputs/visualizations"
LOG_PATH       = "outputs/training_log.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,        exist_ok=True)
os.makedirs("outputs",      exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# One-epoch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """
    Run one epoch (train or eval).
    Returns (mean_loss, accuracy_percent).
    """
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    mean_loss = total_loss / total
    accuracy  = 100.0 * correct / total
    return mean_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device | None = None,
    checkpoint_name: str = "best_model",
) -> dict:
    """
    Train the model and return a dict of per-epoch history.

    Parameters
    ----------
    model          : instantiated HybridSpectralNet (already on device)
    train_loader   : training DataLoader
    val_loader     : validation DataLoader
    num_epochs     : maximum epochs (early stopping may terminate earlier)
    learning_rate  : initial Adam learning rate
    weight_decay   : L2 regularisation strength
    patience       : early-stopping patience (epochs without val improvement)
    device         : torch.device (auto-detected if None)
    checkpoint_name: filename stem for saved checkpoint

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss',
                              'train_acc',  'val_acc'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Training]  device={device}  epochs={num_epochs}  lr={learning_rate}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Cosine annealing: smoothly decays LR to near zero over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_acc   = -1.0
    patience_count = 0
    ckpt_path      = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")

    # CSV log header
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "train_acc", "val_acc", "lr"])

    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} "
          f"{'Train Acc':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        # ── train ──
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        # ── validate ──
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── log ──
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                             f"{train_acc:.2f}", f"{val_acc:.2f}",
                             f"{current_lr:.6f}"])

        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} "
              f"{train_acc:>9.2f}% {val_acc:>8.2f}%  {current_lr:.2e}")

        # ── checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[Early stopping] No improvement for {patience} epochs. "
                      f"Best val acc: {best_val_acc:.2f}%")
                break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  "
          f"Best val acc = {best_val_acc:.2f}%")
    print(f"Best model saved → {ckpt_path}")

    # ── plot curves ──
    _plot_curves(history)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Curve plotting
# ─────────────────────────────────────────────────────────────────────────────
def _plot_curves(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data        import load_dataset
    from step03_preprocess       import preprocess
    from step04_patch_extraction import extract_patches
    from step05_split_dataset    import split_dataset, make_dataloaders
    from step06_model            import HybridSpectralNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y = load_dataset("IndianPines")
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_p, y_l = extract_patches(X_norm, y_remap, patch_size=7)

    X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X_p, y_l)
    train_loader, val_loader, test_loader = make_dataloaders(
        X_tr, X_v, X_te, y_tr, y_v, y_te, batch_size=64
    )

    num_bands   = X_norm.shape[2]
    num_classes = int(y_l.max()) + 1

    model = HybridSpectralNet(
        num_bands=num_bands, num_classes=num_classes, patch_size=7
    ).to(device)

    history = train(
        model, train_loader, val_loader,
        num_epochs=5, learning_rate=1e-3, patience=15, device=device
    )
