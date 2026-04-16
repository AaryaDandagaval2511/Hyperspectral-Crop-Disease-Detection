"""
STEP 12 — Cross-Sensor Testing (Sentinel-2 ↔ Hyperspectral)
=============================================================
Goal: Evaluate whether a model trained on a hyperspectral dataset
      (Indian Pines / Salinas) can classify the same land-cover / crop
      types when given only the 10–13 bands available in Sentinel-2.

The mismatch problem
---------------------
  Hyperspectral dataset  →  200 bands  (approx. 400–2500 nm, 10 nm spacing)
  Sentinel-2             →  10–13 bands (discrete, 10–60 m resolution)

Strategy (Spectral Band Mapping)
----------------------------------
  1.  Map each Sentinel-2 band to the nearest hyperspectral band index
      using the known central wavelength of each sensor.
  2.  Extract only those ~10 hyperspectral bands from the benchmark cube
      so that the spectral dimension matches Sentinel-2.
  3.  Retrain the 3D CNN on the reduced-band benchmark cube
      (this model simulates what you would have trained on Sentinel-2).
  4.  (Optional) Apply the same trained model to the actual Sentinel-2
      cube (step 11) by extracting patches from it in the same way.
  5.  Compare OA / AA / Kappa: full-band model vs cross-sensor model.

Sentinel-2 band wavelengths used (L2A, central wavelengths in nm)
-------------------------------------------------------------------
  B02 490, B03 560, B04 665, B05 705, B06 740, B07 783,
  B08 842, B8A 865, B11 1610, B12 2190

Indian Pines wavelength range: ~400–2500 nm  (200 corrected bands)
  Approximate spacing = (2500 - 400) / 200 ≈ 10.5 nm per band
  Band index for wavelength λ ≈ (λ - 400) / 10.5

Outputs
--------
  outputs/cross_sensor_report.txt
  outputs/visualizations/cross_sensor_oa_comparison.png
"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ_DIR = "outputs/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset wavelength metadata
# (adjust start_nm / end_nm / num_bands for your actual dataset version)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_WAVELENGTHS = {
    "IndianPines": {"start_nm": 400,  "end_nm": 2500, "num_bands": 200},
    "Salinas":     {"start_nm": 400,  "end_nm": 2500, "num_bands": 200},
    "PaviaU":      {"start_nm": 430,  "end_nm": 860,  "num_bands": 103},
}

# Sentinel-2 L2A central wavelengths (nm)
SENTINEL2_WAVELENGTHS_NM = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
SENTINEL2_BAND_IDS        = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]


def wavelength_to_band_index(
    wavelength_nm: float,
    start_nm: float,
    end_nm: float,
    num_bands: int,
) -> int:
    """
    Convert a physical wavelength (nm) to a 0-indexed band number for a
    uniformly-spaced hyperspectral sensor.

    Parameters
    ----------
    wavelength_nm : target wavelength to look up
    start_nm      : first band centre wavelength
    end_nm        : last  band centre wavelength
    num_bands     : total number of bands in the hyperspectral dataset

    Returns
    -------
    band_index : int  (clipped to [0, num_bands-1])
    """
    spacing = (end_nm - start_nm) / (num_bands - 1)
    idx     = round((wavelength_nm - start_nm) / spacing)
    return int(np.clip(idx, 0, num_bands - 1))


def get_sentinel2_equivalent_bands(dataset_name: str) -> list[int]:
    """
    Return the list of hyperspectral band indices that best correspond to
    each Sentinel-2 band for a given benchmark dataset.
    """
    meta = DATASET_WAVELENGTHS.get(dataset_name)
    if meta is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         "Add its wavelength metadata to DATASET_WAVELENGTHS.")

    band_indices = []
    print(f"\n[Cross-Sensor] Sentinel-2 → {dataset_name} band mapping")
    print(f"{'S2 Band':>8}  {'λ (nm)':>8}  {'HSI Index':>10}  {'HSI λ (nm)':>12}")
    print("-" * 46)

    spacing = (meta["end_nm"] - meta["start_nm"]) / (meta["num_bands"] - 1)

    for s2_id, wl in zip(SENTINEL2_BAND_IDS, SENTINEL2_WAVELENGTHS_NM):
        hsi_idx  = wavelength_to_band_index(
            wl, meta["start_nm"], meta["end_nm"], meta["num_bands"]
        )
        hsi_wl   = meta["start_nm"] + hsi_idx * spacing
        band_indices.append(hsi_idx)
        print(f"{s2_id:>8}  {wl:>8}  {hsi_idx:>10}  {hsi_wl:>12.1f}")

    # remove duplicates (e.g. Pavia University only covers 430–860 nm;
    # SWIR bands B11 and B12 both map to the last band)
    unique_indices = list(dict.fromkeys(band_indices))   # preserves order
    print(f"\nUnique band indices: {unique_indices}  ({len(unique_indices)} bands)")
    return unique_indices


def select_bands(
    X_patches: np.ndarray,   # (N, P, P, Bands)
    band_indices: list[int],
) -> np.ndarray:
    """
    Subset a patch array to only the specified spectral band indices.

    Returns
    -------
    X_reduced : (N, P, P, len(band_indices))
    """
    return X_patches[:, :, :, band_indices]


def cross_sensor_experiment(
    X_patches_train: np.ndarray,
    X_patches_val:   np.ndarray,
    X_patches_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    dataset_name: str,
    patch_size: int,
    num_classes: int,
    device: torch.device,
    num_epochs: int = 5,
    batch_size: int = 64,
) -> dict:
    """
    Train and evaluate a 3D CNN using only Sentinel-2-equivalent bands.

    Returns
    -------
    metrics : {"OA": float, "AA": float, "Kappa": float}
    """
    from step05_split_dataset import make_dataloaders
    from step06_model import HybridSpectralNet
    from step07_train import train
    from step08_evaluate import evaluate

    band_indices = get_sentinel2_equivalent_bands(dataset_name)
    n_bands      = len(band_indices)

    Xtr_r = select_bands(X_patches_train, band_indices)
    Xva_r = select_bands(X_patches_val,   band_indices)
    Xte_r = select_bands(X_patches_test,  band_indices)

    print(f"\nReduced patch shape: {Xtr_r.shape}  ({n_bands} Sentinel-2-equivalent bands)")

    tr_l, va_l, te_l = make_dataloaders(
        Xtr_r, Xva_r, Xte_r,
        y_train, y_val, y_test,
        batch_size=batch_size,
    )

    model = HybridSpectralNet(
        num_bands=n_bands, num_classes=num_classes, patch_size=patch_size
    ).to(device)

    train(
        model, tr_l, va_l,
        num_epochs=num_epochs, learning_rate=1e-3, patience=10,
        device=device,
        checkpoint_name=f"cross_sensor_{dataset_name}",
    )

    metrics = evaluate(
        model, te_l, device,
        dataset_name=f"{dataset_name}_S2equiv",
    )
    return {"OA": metrics["OA"], "AA": metrics["AA"], "Kappa": metrics["Kappa"]}


def plot_comparison(
    results: dict,   # {"Full Band": {"OA":..}, "Cross-Sensor":{"OA":..}, ...}
    dataset_name: str,
) -> None:
    """Bar chart comparing OA / AA / Kappa for different configurations."""
    labels     = list(results.keys())
    oa_values  = [results[k]["OA"]    for k in labels]
    aa_values  = [results[k]["AA"]    for k in labels]
    kap_values = [100 * results[k]["Kappa"] for k in labels]  # scale to %

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, oa_values,  width, label="OA (%)",       color="steelblue")
    ax.bar(x,         aa_values,  width, label="AA (%)",       color="darkorange")
    ax.bar(x + width, kap_values, width, label="Kappa × 100",  color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{dataset_name} — Full-Band vs Cross-Sensor Comparison",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(VIZ_DIR, f"{dataset_name}_cross_sensor_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def write_cross_sensor_report(results: dict, dataset_name: str) -> None:
    path = "outputs/cross_sensor_report.txt"
    with open(path, "w") as f:
        f.write(f"Cross-Sensor Evaluation Report — {dataset_name}\n\n")
        f.write(f"{'Configuration':25}  {'OA (%)':>8}  {'AA (%)':>8}  {'Kappa':>8}\n")
        f.write("-" * 55 + "\n")
        for cfg, m in results.items():
            f.write(f"{cfg:25}  {m['OA']:>8.2f}  {m['AA']:>8.2f}  {m['Kappa']:>8.4f}\n")
    print(f"[saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from step01_load_data        import load_dataset
    from step03_preprocess       import preprocess
    from step04_patch_extraction import extract_patches
    from step05_split_dataset    import split_dataset

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET    = "IndianPines"
    PATCH_SIZE = 7

    X, y = load_dataset(DATASET)
    X_norm, y_remap, _, _ = preprocess(X, y)
    X_p, y_l = extract_patches(X_norm, y_remap, patch_size=PATCH_SIZE)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X_p, y_l)

    num_classes = int(y_l.max()) + 1

    # ── evaluate cross-sensor model ──────────────────────────────────────────
    cs_metrics = cross_sensor_experiment(
        X_tr, X_v, X_te, y_tr, y_v, y_te,
        dataset_name=DATASET,
        patch_size=PATCH_SIZE,
        num_classes=num_classes,
        device=device,
        num_epochs=40,
    )

    # ── load the full-band result from step 8 to compare ────────────────────
    # (replace with the actual values printed during step 8)
    FULL_BAND_OA    = 99.85
    FULL_BAND_AA    = 99.88
    FULL_BAND_KAPPA = 0.9983

    comparison = {
        "Full Band (200)":    {"OA": FULL_BAND_OA,   "AA": FULL_BAND_AA,
                               "Kappa": FULL_BAND_KAPPA},
        f"S2-equiv ({len(get_sentinel2_equivalent_bands(DATASET))} bands)":
                              cs_metrics,
    }

    plot_comparison(comparison, DATASET)
    write_cross_sensor_report(comparison, DATASET)

    print("\n--- Cross-Sensor Summary ---")
    for cfg, m in comparison.items():
        print(f"  {cfg:30}  OA={m['OA']:.2f}%  AA={m['AA']:.2f}%  "
              f"Kappa={m['Kappa']:.4f}")
