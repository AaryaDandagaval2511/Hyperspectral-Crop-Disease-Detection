"""
STEP 1 — Load Hyperspectral Data
=================================
Loads .mat files for Indian Pines, Salinas, and Pavia University datasets.
Each dataset contains a hyperspectral cube (H x W x Bands) and a ground truth
label map (H x W) where 0 = unlabeled pixel.

Expected file structure:
    data/
        Indian_pines_corrected.mat
        Indian_pines_gt.mat
        Salinas_corrected.mat
        Salinas_gt.mat
        PaviaU.mat
        PaviaU_gt.mat
"""

import os
import numpy as np
import scipy.io as sio


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these paths to match your actual folder structure
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"   # folder containing your .mat files

DATASETS = {
    "IndianPines": {
        "data_file":  "Indian_pines_corrected.mat",
        "gt_file":    "Indian_pines_gt.mat",
        # common key names inside the .mat files (we try all of them)
        "data_keys":  ["indian_pines_corrected", "data"],
        "gt_keys":    ["indian_pines_gt", "gt", "groundtruth"],
    },
    "Salinas": {
        "data_file":  "Salinas_corrected.mat",
        "gt_file":    "Salinas_gt.mat",
        "data_keys":  ["salinas_corrected", "data"],
        "gt_keys":    ["salinas_gt", "gt", "groundtruth"],
    },
    "PaviaU": {
        "data_file":  "PaviaU.mat",
        "gt_file":    "PaviaU_gt.mat",
        "data_keys":  ["paviaU", "pavia_university", "data"],
        "gt_keys":    ["paviaU_gt", "pavia_university_gt", "gt", "groundtruth"],
    },
}


def _load_mat_key(mat_dict: dict, candidate_keys: list) -> np.ndarray:
    """Try each candidate key; return the first match. Raise if none found."""
    # strip private scipy keys that start with '__'
    available = {k: v for k, v in mat_dict.items() if not k.startswith("__")}
    for key in candidate_keys:
        if key in available:
            return available[key].astype(np.float32)
    # fallback: return the first non-private value
    for v in available.values():
        if isinstance(v, np.ndarray):
            return v.astype(np.float32)
    raise KeyError(
        f"Could not find data in .mat file. Available keys: {list(available.keys())}"
    )


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single dataset by name (e.g. 'IndianPines').

    Returns
    -------
    X : np.ndarray  shape (H, W, Bands)  — float32, raw reflectance values
    y : np.ndarray  shape (H, W)         — int32, class labels (0 = unlabeled)
    """
    cfg = DATASETS[name]

    data_path = os.path.join(DATA_DIR, cfg["data_file"])
    gt_path   = os.path.join(DATA_DIR, cfg["gt_file"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found:   {gt_path}")

    data_mat = sio.loadmat(data_path)
    gt_mat   = sio.loadmat(gt_path)

    X = _load_mat_key(data_mat, cfg["data_keys"])        # (H, W, Bands)
    y = _load_mat_key(gt_mat,   cfg["gt_keys"]).astype(np.int32)  # (H, W)

    # squeeze any extra singleton dimensions (some .mat files have shape (H,W,1,B))
    X = X.squeeze()
    y = y.squeeze()

    # ensure X is 3-D (H, W, Bands)
    if X.ndim == 2:                # single-band edge case
        X = X[:, :, np.newaxis]

    print(f"\n{'='*55}")
    print(f"Dataset     : {name}")
    print(f"Data shape  : {X.shape}  (H x W x Bands)")
    print(f"Labels shape: {y.shape}  (H x W)")
    print(f"Num classes : {len(np.unique(y)) - 1}  (excluding background 0)")
    print(f"Value range : [{X.min():.2f}, {X.max():.2f}]")
    labeled_px  = np.sum(y > 0)
    total_px    = y.size
    print(f"Labeled px  : {labeled_px} / {total_px} ({100*labeled_px/total_px:.1f}%)")

    return X, y


def load_all_datasets() -> dict:
    """Load all three benchmark datasets and return them as a dict."""
    results = {}
    for name in DATASETS:
        try:
            X, y = load_dataset(name)
            results[name] = {"X": X, "y": y}
        except FileNotFoundError as e:
            print(f"[SKIP] {name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = load_all_datasets()
    print(f"\nSuccessfully loaded: {list(datasets.keys())}")
