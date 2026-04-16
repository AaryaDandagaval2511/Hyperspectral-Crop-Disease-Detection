"""
STEP 9 — Band Reduction via PCA
=================================
Applies Principal Component Analysis to the spectral dimension to reduce
hundreds of bands down to a much smaller set of principal components (PCs).

Why this matters for your research
------------------------------------
  • Reduces training time and memory significantly.
  • Forces the model to rely on the most informative spectral directions.
  • Comparing accuracy vs. PC count is a key contribution of your work.

Pipeline
--------
  1. Fit PCA on all pixels of the *training* data  (avoid data leakage).
  2. Transform train / val / test cubes.
  3. Train a fresh model on the reduced cube.
  4. Compare OA / AA / Kappa with the full-band model (step 8).
  5. Plot: Explained Variance vs Component Count.
  6. Plot: OA vs Number of PCs.

Output
------
  outputs/visualizations/pca_explained_variance.png
  outputs/visualizations/pca_oa_vs_ncomponents.png
  outputs/pca_comparison_report.txt
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

VIZ_DIR = "outputs/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PCA fitting and transform
# ─────────────────────────────────────────────────────────────────────────────
def fit_pca(
    X_patches_train: np.ndarray,
    n_components: int,
) -> PCA:
    """
    Fit PCA on the *centre pixel* of training patches only (avoids leakage).

    Parameters
    ----------
    X_patches_train : (N_train, P, P, Bands)  float32
    n_components    : number of PCA components to keep

    Returns
    -------
    fitted sklearn PCA object
    """
    P   = X_patches_train.shape[1]
    mid = P // 2
    # extract just the centre pixel from every patch → (N_train, Bands)
    centre_pixels = X_patches_train[:, mid, mid, :]
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(centre_pixels)
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"[PCA] n_components={n_components}  "
          f"explained variance={explained:.2f}%")
    return pca


def apply_pca_to_patches(
    X_patches: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """
    Apply a *fitted* PCA to all spatial positions in every patch.

    Transforms (N, P, P, Bands) → (N, P, P, n_components).
    Each pixel in every patch is projected independently.
    """
    N, P, P2, B = X_patches.shape
    # reshape to (N*P*P, Bands), transform, reshape back
    pixels   = X_patches.reshape(-1, B)                        # (N*P*P, B)
    reduced  = pca.transform(pixels).astype(np.float32)        # (N*P*P, k)
    k        = reduced.shape[1]
    out      = reduced.reshape(N, P, P, k)                     # (N, P, P, k)
    return out


def apply_pca_to_cube(
    X_cube: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Apply PCA to a full (H, W, Bands) cube → (H, W, n_components)."""
    H, W, B = X_cube.shape
    pixels  = X_cube.reshape(-1, B)
    reduced = pca.transform(pixels).astype(np.float32)
    return reduced.reshape(H, W, pca.n_components_)


# ─────────────────────────────────────────────────────────────────────────────
# Explained variance curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_explained_variance(
    X_patches_train: np.ndarray,
    dataset_name: str,
    max_components: int | None = None,
) -> None:
    """
    Fit PCA with all components and plot cumulative explained variance.
    Draws a horizontal line at 95% and 99% thresholds.
    """
    P   = X_patches_train.shape[1]
    mid = P // 2
    centre = X_patches_train[:, mid, mid, :]
    B      = centre.shape[1]
    n_max  = min(max_components or B, B, len(centre) - 1)

    pca_full = PCA(n_components=n_max, random_state=42)
    pca_full.fit(centre)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    idx_95 = np.searchsorted(cumvar, 95) + 1
    idx_99 = np.searchsorted(cumvar, 99) + 1

    print(f"[PCA] Components for 95% variance: {idx_95}")
    print(f"[PCA] Components for 99% variance: {idx_99}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2)
    ax.axhline(95, color="orange", linestyle="--", label=f"95% ({idx_95} PCs)")
    ax.axhline(99, color="red",    linestyle="--", label=f"99% ({idx_99} PCs)")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title(f"{dataset_name} — PCA Explained Variance", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, f"{dataset_name}_pca_explained_variance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    return idx_95, idx_99


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy vs PC count experiment
# ─────────────────────────────────────────────────────────────────────────────
def pca_experiment(
    X_patches_train: np.ndarray,
    X_patches_val:   np.ndarray,
    X_patches_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    pc_counts: list[int],
    patch_size: int,
    num_classes: int,
    device: torch.device,
    dataset_name: str = "dataset",
    num_epochs: int = 40,
    batch_size: int = 64,
) -> dict:
    """
    For each value in pc_counts: apply PCA, retrain, evaluate.

    Returns
    -------
    results : {n_pc: {"OA": float, "AA": float, "Kappa": float}}
    """
    from step05_split_dataset import make_dataloaders
    from step06_model import HybridSpectralNet
    from step07_train import train
    from step08_evaluate import evaluate

    results = {}

    for n_pc in pc_counts:
        print(f"\n{'─'*55}")
        print(f" PCA experiment: n_components = {n_pc}")
        print(f"{'─'*55}")

        pca = fit_pca(X_patches_train, n_components=n_pc)

        Xtr_r = apply_pca_to_patches(X_patches_train, pca)
        Xva_r = apply_pca_to_patches(X_patches_val,   pca)
        Xte_r = apply_pca_to_patches(X_patches_test,  pca)

        tr_l, va_l, te_l = make_dataloaders(
            Xtr_r, Xva_r, Xte_r,
            y_train, y_val, y_test,
            batch_size=batch_size,
        )

        model = HybridSpectralNet(
            num_bands=n_pc, num_classes=num_classes, patch_size=patch_size
        ).to(device)

        train(
            model, tr_l, va_l,
            num_epochs=num_epochs, learning_rate=1e-3, patience=10,
            device=device,
            checkpoint_name=f"pca_{n_pc}pc",
        )

        metrics = evaluate(
            model, te_l, device,
            dataset_name=f"{dataset_name}_pca{n_pc}pc",
        )
        results[n_pc] = {
            "OA":    metrics["OA"],
            "AA":    metrics["AA"],
            "Kappa": metrics["Kappa"],
        }

    # ── plot OA vs n_pc ──
    pcs = sorted(results.keys())
    oas = [results[p]["OA"] for p in pcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pcs, oas, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title(f"{dataset_name} — OA vs PCA Components", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, f"{dataset_name}_pca_oa_vs_npc.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

    # ── text report ──
    rpt_path = "outputs/pca_comparison_report.txt"
    with open(rpt_path, "w") as f:
        f.write(f"PCA Band Reduction — {dataset_name}\n\n")
        f.write(f"{'n_PCs':>8}  {'OA (%)':>8}  {'AA (%)':>8}  {'Kappa':>8}\n")
        f.write("-" * 40 + "\n")
        for p in pcs:
            r = results[p]
            f.write(f"{p:>8}  {r['OA']:>8.2f}  {r['AA']:>8.2f}  {r['Kappa']:>8.4f}\n")
    print(f"[saved] {rpt_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data        import load_dataset
    from step03_preprocess       import preprocess
    from step04_patch_extraction import extract_patches
    from step05_split_dataset    import split_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET    = "IndianPines"
    PATCH_SIZE = 7

    X, y = load_dataset(DATASET)
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_p, y_l = extract_patches(X_norm, y_remap, patch_size=PATCH_SIZE)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X_p, y_l)

    # plot explained variance curve first
    plot_explained_variance(X_tr, DATASET)

    # run experiment with several PC counts
    pca_experiment(
        X_tr, X_v, X_te, y_tr, y_v, y_te,
        pc_counts=[10, 20, 30, 50],
        patch_size=PATCH_SIZE,
        num_classes=int(y_l.max()) + 1,
        device=device,
        dataset_name=DATASET,
        num_epochs=10,
    )
