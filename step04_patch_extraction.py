"""
STEP 4 — Patch Extraction
===========================
Converts the hyperspectral cube (H × W × Bands) into a dataset of small
spatial patches.  Each patch is centred on a labeled pixel.

  patch shape : (patch_size, patch_size, Bands)
  label       : integer class of the centre pixel

The cube is zero-padded by (patch_size // 2) on every side so that pixels
near the image border still produce full-size patches.

Output
------
  X_patches : np.ndarray  (N, patch_size, patch_size, Bands)  float32
  y_labels  : np.ndarray  (N,)                                int64
where N = number of labeled pixels.
"""

import numpy as np


def extract_patches(
    X_norm: np.ndarray,
    y_remap: np.ndarray,
    patch_size: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (patch_size × patch_size × Bands) patches centred on every
    labeled pixel in the hyperspectral cube.

    Parameters
    ----------
    X_norm     : (H, W, Bands)  normalised hyperspectral cube
    y_remap    : (H, W)         label map (0 = background, ≥1 = class)
    patch_size : spatial size of each patch (must be odd; common: 5, 7, 9, 11)

    Returns
    -------
    X_patches  : (N, patch_size, patch_size, Bands)
    y_labels   : (N,)  int64
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd (e.g. 5, 7, 9, 11).")

    H, W, B = X_norm.shape
    half    = patch_size // 2

    # ── zero-pad the cube so border pixels get full patches ──────────────────
    # pad only the spatial dimensions (axis 0 and 1), not spectral (axis 2)
    X_padded = np.pad(
        X_norm,
        pad_width=((half, half), (half, half), (0, 0)),
        mode="constant",
        constant_values=0,
    )  # shape: (H + 2*half, W + 2*half, Bands)

    # ── collect labeled pixel positions ──────────────────────────────────────
    rows, cols = np.where(y_remap > 0)   # only labeled pixels
    N = len(rows)

    X_patches = np.empty((N, patch_size, patch_size, B), dtype=np.float32)
    y_labels  = np.empty(N, dtype=np.int64)

    for i, (r, c) in enumerate(zip(rows, cols)):
        # in padded coords the pixel is at (r + half, c + half)
        pr, pc = r + half, c + half
        patch  = X_padded[pr - half : pr + half + 1,
                          pc - half : pc + half + 1, :]   # (P, P, B)
        X_patches[i] = patch
        y_labels[i]  = y_remap[r, c]

    # ── shift labels to 0-indexed for CrossEntropyLoss ────────────────────────
    # labels come in as 1..num_classes; subtract 1 to get 0..num_classes-1
    y_labels = y_labels - 1

    print(f"[Patch Extraction]  patch_size={patch_size}")
    print(f"  X_patches shape : {X_patches.shape}  "
          f"(N={N}, P={patch_size}, P={patch_size}, B={B})")
    print(f"  y_labels  shape : {y_labels.shape}")
    print(f"  label range     : [{y_labels.min()}, {y_labels.max()}]  "
          f"(0-indexed, {y_labels.max()+1} classes)")
    print(f"  memory (approx) : {X_patches.nbytes / 1e6:.1f} MB")

    return X_patches, y_labels


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data import load_dataset
    from step03_preprocess import preprocess

    X, y = load_dataset("IndianPines")
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_patches, y_labels   = extract_patches(X_norm, y_remap, patch_size=7)
    print("\nSample patch min/max:", X_patches[0].min(), X_patches[0].max())
