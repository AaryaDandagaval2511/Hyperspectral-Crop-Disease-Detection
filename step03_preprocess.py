"""
STEP 3 — Preprocessing
========================
Two normalisation strategies are offered:
  • 'minmax'   — scales each band to [0, 1]   (fast, simple)
  • 'standard' — zero-mean / unit-variance per band (often better for CNN)

Unlabeled pixels (label == 0) are retained in the cube but excluded when
building the patch dataset in Step 4.

Both the normalised cube and the (optionally filtered) label map are returned
as NumPy arrays ready for patch extraction.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_cube(
    X: np.ndarray,
    method: str = "standard",
) -> np.ndarray:
    """
    Normalise a hyperspectral cube band-by-band.

    Parameters
    ----------
    X      : (H, W, Bands)  raw cube
    method : 'minmax'  → each band scaled to [0, 1]
             'standard' → each band zero-mean / unit-variance

    Returns
    -------
    X_norm : (H, W, Bands)  float32 normalised cube
    """
    H, W, B = X.shape
    X_2d = X.reshape(-1, B)          # (H*W, Bands)  — reshape for sklearn

    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'minmax' or 'standard'.")

    X_scaled = scaler.fit_transform(X_2d)   # (H*W, Bands)
    X_norm   = X_scaled.reshape(H, W, B).astype(np.float32)

    print(f"[Normalize] method={method}  "
          f"value range after: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    return X_norm


def get_labeled_mask(y: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of shape (H, W) where True = labeled pixel (label > 0).

    We never modify or drop pixels from the cube — the mask is only used
    later when building the patch dataset so we skip patches centred on
    unlabeled pixels.
    """
    mask = y > 0
    total    = y.size
    labeled  = mask.sum()
    print(f"[Labeled mask]  labeled={labeled} / {total}  "
          f"({100*labeled/total:.1f}%)")
    return mask


def remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Remap class labels so that they form a contiguous range starting at 0,
    ignoring the background class (original label 0).

    Example: if the dataset has classes {0,1,2,3,9,11} the mapping will be
        0→0 (background kept as 0), 1→1, 2→2, 3→3, 9→4, 11→5

    Returns
    -------
    y_remap  : (H, W) int32 with new labels
    mapping  : dict  {original_label: new_label}
    """
    unique = np.unique(y)
    # 0 stays 0 (background); remap everything else to 1..N
    mapping = {0: 0}
    counter = 1
    for lbl in unique:
        if lbl != 0:
            mapping[lbl] = counter
            counter += 1
    y_remap = np.vectorize(mapping.get)(y).astype(np.int32)
    print(f"[Remap labels]  original classes: {list(unique)}")
    print(f"                new classes     : {list(np.unique(y_remap))}")
    return y_remap, mapping


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    norm_method: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Full preprocessing pipeline for one dataset.

    Parameters
    ----------
    X           : (H, W, Bands) raw float32 cube
    y           : (H, W) int32 ground truth labels
    norm_method : 'minmax' or 'standard'

    Returns
    -------
    X_norm      : (H, W, Bands) normalised cube
    y_remap     : (H, W) remapped labels (0 = background, 1..N = classes)
    labeled_mask: (H, W) boolean mask
    label_map   : dict {original → new} label mapping
    """
    print("\n--- Preprocessing ---")
    X_norm             = normalize_cube(X, method=norm_method)
    y_remap, label_map = remap_labels(y)
    labeled_mask       = get_labeled_mask(y_remap)
    return X_norm, y_remap, labeled_mask, label_map


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data import load_dataset

    X, y   = load_dataset("IndianPines")
    X_norm, y_remap, mask, lmap = preprocess(X, y, norm_method="standard")
    print("\nX_norm shape :", X_norm.shape)
    print("y_remap shape:", y_remap.shape)
    print("Label map    :", lmap)
