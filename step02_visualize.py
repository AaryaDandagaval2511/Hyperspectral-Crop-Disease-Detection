"""
STEP 2 — Data Visualization
=============================
Visualizes:
  • Several individual spectral band images (grayscale)
  • A false-colour composite (RGB from three chosen bands)
  • The ground-truth label map with a colour-coded legend

All figures are saved to  outputs/visualizations/  as PNG files.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

OUTPUT_DIR = "outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── class-name look-up tables ───────────────────────────────────────────────
CLASS_NAMES = {
    "IndianPines": [
        "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
        "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
        "Soybean-clean", "Wheat", "Woods",
        "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
    ],
    "Salinas": [
        "Background", "Broccoli_green_weeds_1", "Broccoli_green_weeds_2",
        "Fallow", "Fallow_rough_plow", "Fallow_smooth", "Stubble",
        "Celery", "Grapes_untrained", "Soil_vinyard_develop",
        "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
        "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
        "Lettuce_romaine_7wk", "Vinyard_untrained",
        "Vinyard_vertical_trellis",
    ],
    "PaviaU": [
        "Background", "Asphalt", "Meadows", "Gravel", "Trees",
        "Painted metal sheets", "Bare Soil", "Bitumen",
        "Self-Blocking Bricks", "Shadows",
    ],
}


def plot_spectral_bands(
    X: np.ndarray,
    dataset_name: str,
    num_bands: int = 6,
) -> None:
    """
    Plot `num_bands` evenly-spaced spectral band images side by side.

    Parameters
    ----------
    X            : hyperspectral cube (H, W, Bands)
    dataset_name : used for title and filename
    num_bands    : how many bands to show (default 6)
    """
    total_bands = X.shape[2]
    indices = np.linspace(0, total_bands - 1, num_bands, dtype=int)

    fig, axes = plt.subplots(1, num_bands, figsize=(3 * num_bands, 3.5))
    fig.suptitle(f"{dataset_name} — Sample Spectral Bands", fontsize=13, fontweight="bold")

    for ax, band_idx in zip(axes, indices):
        band_img = X[:, :, band_idx]
        # min-max normalise for display only
        vmin, vmax = band_img.min(), band_img.max()
        if vmax > vmin:
            band_img = (band_img - vmin) / (vmax - vmin)
        ax.imshow(band_img, cmap="gray")
        ax.set_title(f"Band {band_idx}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_spectral_bands.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_false_colour(
    X: np.ndarray,
    dataset_name: str,
    r_band: int | None = None,
    g_band: int | None = None,
    b_band: int | None = None,
) -> None:
    """
    Create a false-colour RGB composite from three chosen bands.
    Defaults to bands at 2/4, 1/4, and 3/4 of the spectral range.
    """
    total_bands = X.shape[2]
    r = r_band if r_band is not None else int(total_bands * 0.75)
    g = g_band if g_band is not None else int(total_bands * 0.50)
    b = b_band if b_band is not None else int(total_bands * 0.25)

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    rgb = np.stack([norm(X[:, :, r]),
                    norm(X[:, :, g]),
                    norm(X[:, :, b])], axis=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title(
        f"{dataset_name} — False Colour\n(R=Band{r}, G=Band{g}, B=Band{b})",
        fontsize=11, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_false_colour.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_ground_truth(y: np.ndarray, dataset_name: str) -> None:
    """
    Display the ground-truth label map with a colour legend.

    Parameters
    ----------
    y            : label array (H, W), integer, 0 = background/unlabeled
    dataset_name : used for title, legend look-up, and filename
    """
    classes = np.unique(y)
    num_classes = len(classes)

    # build a discrete colourmap with one colour per unique label
    cmap_raw = plt.get_cmap("tab20", num_classes)
    cmap = ListedColormap([cmap_raw(i) for i in range(num_classes)])

    # re-index labels to 0..N-1 for the colourmap
    label_remap = {cls: i for i, cls in enumerate(classes)}
    y_remap = np.vectorize(label_remap.get)(y)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(y_remap, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax.set_title(f"{dataset_name} — Ground Truth Labels",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    # build legend patches
    names = CLASS_NAMES.get(dataset_name, [f"Class {i}" for i in range(100)])
    patches = []
    for i, cls in enumerate(classes):
        label_name = names[cls] if cls < len(names) else f"Class {cls}"
        pixel_count = np.sum(y == cls)
        patch = mpatches.Patch(
            color=cmap_raw(i),
            label=f"{label_name} ({pixel_count} px)",
        )
        patches.append(patch)

    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        frameon=False,
    )
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_ground_truth.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_spectral_signature(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    max_classes: int = 5,
) -> None:
    """
    Plot the mean spectral signature (reflectance vs band index) for
    several classes — useful for understanding class separability.
    """
    classes = [c for c in np.unique(y) if c != 0][:max_classes]
    names   = CLASS_NAMES.get(dataset_name, [f"Class {i}" for i in range(100)])

    fig, ax = plt.subplots(figsize=(9, 4))
    for cls in classes:
        mask    = y == cls
        pixels  = X[mask]            # (N_pixels, Bands)
        mean_sig = pixels.mean(axis=0)
        label = names[cls] if cls < len(names) else f"Class {cls}"
        ax.plot(mean_sig, label=label)

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Mean Reflectance")
    ax.set_title(f"{dataset_name} — Mean Spectral Signatures", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_spectral_signatures.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def visualize_dataset(X: np.ndarray, y: np.ndarray, dataset_name: str) -> None:
    """Run all four visualisation functions for one dataset."""
    print(f"\n--- Visualizing {dataset_name} ---")
    plot_spectral_bands(X, dataset_name)
    plot_false_colour(X, dataset_name)
    plot_ground_truth(y, dataset_name)
    plot_spectral_signature(X, y, dataset_name)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # import loader from step 1
    from step01_load_data import load_all_datasets
    datasets = load_all_datasets()
    for name, d in datasets.items():
        visualize_dataset(d["X"], d["y"], name)
    print("\nAll visualizations saved to:", OUTPUT_DIR)
