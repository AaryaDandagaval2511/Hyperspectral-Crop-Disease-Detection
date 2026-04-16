"""
STEP 10 — Explainability (SHAP + Gradient-based Band Importance)
==================================================================
Two complementary explainability methods are implemented:

Method A — SHAP GradientExplainer
  • Uses back-propagation to estimate each *input feature's* contribution
    to the model output. Works directly on the raw patch tensors.
  • We aggregate SHAP values over the spatial P×P dimensions to get a
    per-band importance score.

Method B — Spectral Gradient Saliency (fallback / fast alternative)
  • Computes |∂loss/∂input| and averages over the spatial patch dimensions
    to get a per-band saliency score. No SHAP library needed.
  • Very fast and interpretable — good for quick diagnosis.

Outputs
-------
  outputs/visualizations/{dataset}_shap_band_importance.png
  outputs/visualizations/{dataset}_gradient_saliency.png
  outputs/shap_band_scores.npy   (raw per-band scores for later use)

Note: SHAP is slow on large test sets.  We sample up to 200 test patches
for the background set and 100 for evaluation to keep runtime reasonable.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ_DIR = "outputs/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Method B — Gradient Saliency (always available, no extra library)
# ─────────────────────────────────────────────────────────────────────────────
def gradient_band_importance(
    model: nn.Module,
    X_patches: np.ndarray,     # (N, P, P, Bands)
    y_labels:  np.ndarray,     # (N,)
    device: torch.device,
    num_samples: int = 500,
    dataset_name: str = "dataset",
) -> np.ndarray:
    """
    Compute mean |gradient| for each spectral band.

    Steps
    -----
    1. Convert patches to tensors (N, 1, Bands, P, P).
    2. Forward + backward pass to get input gradients.
    3. Average |gradient| over (N, 1, P, P) leaving shape (Bands,).

    Returns
    -------
    band_scores : (Bands,) float32  — higher = more important
    """
    model.eval()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_patches), size=min(num_samples, len(X_patches)),
                     replace=False)
    X_sub = X_patches[idx]    # (M, P, P, Bands)
    y_sub = y_labels[idx]

    # tensor shape (M, 1, Bands, P, P)
    X_t = torch.from_numpy(
        X_sub.transpose(0, 3, 1, 2)[:, np.newaxis]
    ).float().to(device)
    X_t.requires_grad_(True)

    y_t = torch.from_numpy(y_sub).long().to(device)

    criterion = nn.CrossEntropyLoss()
    logits    = model(X_t)
    loss      = criterion(logits, y_t)
    loss.backward()

    # grad shape: (M, 1, Bands, P, P)
    grad = X_t.grad.abs().detach().cpu().numpy()
    # average over batch, channel, and spatial dims → (Bands,)
    band_scores = grad.mean(axis=(0, 1, 3, 4))

    _plot_band_importance(
        band_scores,
        title=f"{dataset_name} — Gradient Band Saliency",
        save_name=f"{dataset_name}_gradient_saliency.png",
    )

    np.save(f"outputs/{dataset_name}_gradient_band_scores.npy", band_scores)
    print(f"[saved] outputs/{dataset_name}_gradient_band_scores.npy")

    return band_scores


# ─────────────────────────────────────────────────────────────────────────────
# Method A — SHAP GradientExplainer
# ─────────────────────────────────────────────────────────────────────────────
def shap_band_importance(
    model: nn.Module,
    X_patches: np.ndarray,     # (N, P, P, Bands)
    device: torch.device,
    background_size: int = 100,
    explain_size: int    = 50,
    dataset_name: str    = "dataset",
) -> np.ndarray | None:
    """
    Compute SHAP values for the spectral bands using GradientExplainer.

    If the `shap` library is not installed this function prints an
    installation hint and returns None (the pipeline continues without SHAP).

    Returns
    -------
    band_scores : (Bands,) float32  or None
    """
    try:
        import shap
    except ImportError:
        print(
            "\n[SHAP] Library not installed.\n"
            "       Install with:  pip install shap\n"
            "       Falling back to gradient saliency only.\n"
        )
        return None

    model.eval()
    rng = np.random.default_rng(0)

    # ── background set (model baseline) ──────────────────────────────────────
    bg_idx = rng.choice(len(X_patches), size=min(background_size, len(X_patches)),
                        replace=False)
    X_bg = X_patches[bg_idx].transpose(0, 3, 1, 2)[:, np.newaxis]  # (M,1,B,P,P)
    X_bg_t = torch.from_numpy(X_bg).float().to(device)

    # ── explain set ───────────────────────────────────────────────────────────
    ex_idx = rng.choice(len(X_patches), size=min(explain_size, len(X_patches)),
                        replace=False)
    X_ex = X_patches[ex_idx].transpose(0, 3, 1, 2)[:, np.newaxis]  # (M,1,B,P,P)
    X_ex_t = torch.from_numpy(X_ex).float().to(device)

    print(f"[SHAP] background={len(bg_idx)}  explain={len(ex_idx)}")

    explainer  = shap.GradientExplainer(model, X_bg_t)
    shap_vals  = explainer.shap_values(X_ex_t)
    # shap_vals: list of length num_classes, each (M, 1, Bands, P, P)

    # mean |SHAP| over all classes, samples, channel, and spatial dims
    shap_arr  = np.stack(shap_vals, axis=0)         # (C, M, 1, B, P, P)
    band_scores = np.abs(shap_arr).mean(axis=(0, 1, 2, 4, 5))  # (B,)

    _plot_band_importance(
        band_scores,
        title=f"{dataset_name} — SHAP Band Importance",
        save_name=f"{dataset_name}_shap_band_importance.png",
    )

    np.save(f"outputs/{dataset_name}_shap_band_scores.npy", band_scores)
    print(f"[saved] outputs/{dataset_name}_shap_band_scores.npy")

    return band_scores


# ─────────────────────────────────────────────────────────────────────────────
# Shared plotting helper
# ─────────────────────────────────────────────────────────────────────────────
def _plot_band_importance(
    scores: np.ndarray,
    title: str,
    save_name: str,
    top_k: int = 20,
) -> None:
    """
    Two-panel plot:
      Left  — bar chart of all bands (importance vs band index)
      Right — bar chart of the top-k most important bands
    """
    B = len(scores)
    top_idx = np.argsort(scores)[::-1][:top_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # all bands
    ax1.bar(range(B), scores, width=1.0, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Band Index")
    ax1.set_ylabel("Importance Score")
    ax1.set_title("All Bands")
    ax1.grid(True, alpha=0.3, axis="y")

    # top-k bands
    top_scores = scores[top_idx]
    ax2.barh(range(top_k)[::-1], top_scores, color="tomato", alpha=0.85)
    ax2.set_yticks(range(top_k)[::-1])
    ax2.set_yticklabels([f"Band {i}" for i in top_idx], fontsize=8)
    ax2.set_xlabel("Importance Score")
    ax2.set_title(f"Top {top_k} Bands")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, save_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def print_top_bands(scores: np.ndarray, top_k: int = 10) -> None:
    """Print the top-k most important band indices and their scores."""
    top_idx = np.argsort(scores)[::-1][:top_k]
    print(f"\nTop {top_k} most important spectral bands:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}.  Band {idx:4d}   score={scores[idx]:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data        import load_dataset
    from step03_preprocess       import preprocess
    from step04_patch_extraction import extract_patches
    from step05_split_dataset    import split_dataset
    from step06_model            import HybridSpectralNet
    from step08_evaluate         import load_best_model

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = "IndianPines"

    X, y = load_dataset(DATASET)
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_p, y_l = extract_patches(X_norm, y_remap, patch_size=7)
    _, _, X_te, _, _, y_te = split_dataset(X_p, y_l)

    num_bands   = X_norm.shape[2]
    num_classes = int(y_l.max()) + 1
    model = HybridSpectralNet(num_bands, num_classes, patch_size=7).to(device)
    model = load_best_model(model, device=device)

    # Method B (always works)
    grad_scores = gradient_band_importance(
        model, X_te, y_te, device, dataset_name=DATASET
    )
    print_top_bands(grad_scores)

    # Method A (needs: pip install shap)
    shap_scores = shap_band_importance(
        model, X_te, device, dataset_name=DATASET
    )
    if shap_scores is not None:
        print_top_bands(shap_scores)
