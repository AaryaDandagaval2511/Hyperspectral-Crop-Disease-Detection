"""
STEP 5 — Train / Validation / Test Split + PyTorch Dataset
============================================================
Splits the patch dataset into train / val / test subsets using a
*stratified* split so that every class is proportionally represented
in each partition.

Default split: 70% train | 10% val | 20% test

Also defines HyperspectralDataset — a torch.utils.data.Dataset that
converts NumPy patches into the tensor format expected by the 3D CNN:

  Input tensor shape (one sample): (Bands, 1, patch_size, patch_size)
      ↑ spectral dim treated as "depth" in the 3D convolution
         ↑ single "channel" dimension required by Conv3d
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HyperspectralDataset(Dataset):
    """
    Wraps (X_patches, y_labels) numpy arrays for use with DataLoader.

    Patch array shape   : (N, patch_size, patch_size, Bands)
    Tensor output shape : (1, Bands, patch_size, patch_size)
        ^ we add a single channel dim so Conv3d sees:
          (batch, 1, depth=Bands, height=P, width=P)
    """

    def __init__(self, patches: np.ndarray, labels: np.ndarray):
        # transpose from (N, P, P, B) → (N, B, P, P) then add channel dim
        # final stored shape: (N, 1, B, P, P)
        self.patches = torch.from_numpy(
            patches.transpose(0, 3, 1, 2)[:, np.newaxis]   # (N,1,B,P,P)
        ).float()
        self.labels  = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Splitting utility
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(
    X_patches: np.ndarray,
    y_labels: np.ndarray,
    test_size: float  = 0.20,
    val_size: float   = 0.10,
    random_state: int = 42,
) -> tuple:
    """
    Stratified split into train / val / test.

    Parameters
    ----------
    X_patches    : (N, P, P, Bands)
    y_labels     : (N,)  0-indexed integer labels
    test_size    : fraction for test   (default 0.20)
    val_size     : fraction for val    (default 0.10)
    random_state : reproducibility seed

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)  — all numpy arrays
    """
    # first split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_patches, y_labels,
        test_size=test_size,
        stratify=y_labels,
        random_state=random_state,
    )

    # then split the remaining data into train + val
    # val fraction relative to the temp set
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_rel,
        stratify=y_temp,
        random_state=random_state,
    )

    print("\n[Split]")
    print(f"  Train : {len(y_train):6d} samples  ({100*len(y_train)/len(y_labels):.1f}%)")
    print(f"  Val   : {len(y_val):6d} samples  ({100*len(y_val)/len(y_labels):.1f}%)")
    print(f"  Test  : {len(y_test):6d} samples  ({100*len(y_test)/len(y_labels):.1f}%)")

    # class distribution check
    classes = np.unique(y_labels)
    print(f"  Num classes: {len(classes)}")
    for split_name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = [np.sum(yy == c) for c in classes]
        print(f"  {split_name} class counts (min/max): "
              f"{min(counts)} / {max(counts)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_dataloaders(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap split arrays in HyperspectralDataset and return DataLoaders.

    Parameters
    ----------
    batch_size   : mini-batch size for training (64–128 is typical)
    num_workers  : number of CPU workers for data loading
                   (set to 0 on Windows or if you hit 'BrokenPipeError')

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = HyperspectralDataset(X_train, y_train)
    val_ds   = HyperspectralDataset(X_val,   y_val)
    test_ds  = HyperspectralDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"\n[DataLoaders]  batch_size={batch_size}")
    print(f"  train batches : {len(train_loader)}")
    print(f"  val   batches : {len(val_loader)}")
    print(f"  test  batches : {len(test_loader)}")

    # show tensor shape for one batch
    x_sample, y_sample = next(iter(train_loader))
    print(f"  input tensor  : {tuple(x_sample.shape)}  "
          f"(batch, 1, Bands, P, P)")
    print(f"  label tensor  : {tuple(y_sample.shape)}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data       import load_dataset
    from step03_preprocess      import preprocess
    from step04_patch_extraction import extract_patches

    X, y = load_dataset("IndianPines")
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_patches, y_labels   = extract_patches(X_norm, y_remap, patch_size=7)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X_patches, y_labels)
    train_loader, val_loader, test_loader = make_dataloaders(
        X_tr, X_v, X_te, y_tr, y_v, y_te, batch_size=64
    )
