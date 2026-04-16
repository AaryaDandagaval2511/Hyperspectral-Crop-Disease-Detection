"""
STEP 8 — Evaluation
=====================
Loads the best checkpoint and evaluates on the held-out test set.

Outputs
-------
• Overall Accuracy (OA)
• Average Accuracy (AA)  — mean per-class accuracy
• Kappa coefficient
• Per-class precision, recall, F1  (sklearn classification_report)
• Confusion matrix  (raw counts + normalised)
• Saved plots: outputs/visualizations/confusion_matrix.png

All metrics are also written to outputs/evaluation_report.txt.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
)

VIZ_DIR    = "outputs/visualizations"
REPORT_DIR = "outputs"
os.makedirs(VIZ_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on all batches and return (y_true, y_pred) arrays.
    """
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(preds)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    dataset_name: str = "dataset",
) -> dict:
    """
    Full evaluation on the test set.

    Parameters
    ----------
    model        : trained model (weights already loaded)
    test_loader  : DataLoader for the test split
    device       : torch.device
    class_names  : list of human-readable class names (optional)
    dataset_name : used for plot titles and filenames

    Returns
    -------
    metrics : dict with OA, AA, Kappa, and per-class F1
    """
    y_true, y_pred = predict(model, test_loader, device)

    # ── scalar metrics ──────────────────────────────────────────────────────
    oa    = 100.0 * accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # per-class accuracy → AA
    classes    = np.unique(y_true)
    per_class  = [
        100.0 * np.sum((y_true == c) & (y_pred == c)) / np.sum(y_true == c)
        for c in classes
    ]
    aa = np.mean(per_class)

    print(f"\n{'='*55}")
    print(f"Evaluation — {dataset_name}")
    print(f"  Overall Accuracy (OA)   : {oa:.2f}%")
    print(f"  Average Accuracy (AA)   : {aa:.2f}%")
    print(f"  Cohen's Kappa           : {kappa:.4f}")
    print(f"{'='*55}")

    # ── per-class report ─────────────────────────────────────────────────────
    if class_names is None:
        target_names = [f"Class {c}" for c in classes]
    else:
        target_names = [class_names[c] if c < len(class_names) else f"Class {c}"
                        for c in classes]

    report = classification_report(
        y_true, y_pred,
        labels=list(classes),
        target_names=target_names,
        digits=4,
    )
    print("\nClassification Report:")
    print(report)

    # ── save text report ─────────────────────────────────────────────────────
    report_path = os.path.join(REPORT_DIR, f"{dataset_name}_eval_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Evaluation — {dataset_name}\n")
        f.write(f"OA    : {oa:.4f}%\n")
        f.write(f"AA    : {aa:.4f}%\n")
        f.write(f"Kappa : {kappa:.4f}\n\n")
        f.write(report)
    print(f"[saved] {report_path}")

    # ── confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    _plot_confusion_matrix(cm, target_names, dataset_name)

    return {
        "OA": oa, "AA": aa, "Kappa": kappa,
        "y_true": y_true, "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    dataset_name: str,
) -> None:
    n = len(class_names)
    fig_size = max(8, n * 0.7)

    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))

    for ax, data, title, fmt in [
        (axes[0], cm, "Counts", "d"),
        (axes[1], cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8),
         "Normalised", ".2f"),
    ]:
        sns.heatmap(
            data, ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", fmt=fmt,
            annot=(n <= 20),       # show numbers only when matrix is small enough
            linewidths=0.5 if n <= 20 else 0,
        )
        ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.suptitle(dataset_name, fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Load best checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_best_model(
    model: nn.Module,
    checkpoint_name: str = "best_model",
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join("outputs/checkpoints", f"{checkpoint_name}.pt")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Loaded checkpoint]  epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.2f}%  ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data        import load_dataset
    from step02_visualize        import CLASS_NAMES
    from step03_preprocess       import preprocess
    from step04_patch_extraction import extract_patches
    from step05_split_dataset    import split_dataset, make_dataloaders
    from step06_model            import HybridSpectralNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = "IndianPines"

    X, y = load_dataset(DATASET)
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_p, y_l = extract_patches(X_norm, y_remap, patch_size=7)
    _, _, X_te, _, _, y_te = split_dataset(X_p, y_l)
    _, _, test_loader = make_dataloaders(
        X_p, X_p, X_te, y_l, y_l, y_te, batch_size=64
    )

    num_bands   = X_norm.shape[2]
    num_classes = int(y_l.max()) + 1

    model = HybridSpectralNet(num_bands, num_classes, patch_size=7).to(device)
    model = load_best_model(model, device=device)

    # build 0-indexed class names (excluding "Background" at index 0)
    names_0idx = CLASS_NAMES.get(DATASET, [])[1:]   # drop background entry
    evaluate(model, test_loader, device, class_names=names_0idx,
             dataset_name=DATASET)
