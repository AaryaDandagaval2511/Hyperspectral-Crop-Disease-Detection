"""
STEP 11 — Sentinel-2 (.SAFE) Processing
=========================================
Loads Sentinel-2 Level-2A data from the standard .SAFE directory format.

Expected folder structure
--------------------------
  S2A_MSIL2A_<date>_<tile>.SAFE/
      GRANULE/
          L2A_T<tile>_<date>/
              IMG_DATA/
                  R10m/
                      *_B02_10m.jp2   ← Blue
                      *_B03_10m.jp2   ← Green
                      *_B04_10m.jp2   ← Red
                      *_B08_10m.jp2   ← NIR
                  R20m/
                      *_B05_20m.jp2   ← Red-Edge 1
                      *_B06_20m.jp2   ← Red-Edge 2
                      *_B07_20m.jp2   ← Red-Edge 3
                      *_B8A_20m.jp2   ← NIR narrow
                      *_B11_20m.jp2   ← SWIR 1
                      *_B12_20m.jp2   ← SWIR 2
                  R60m/
                      *_B01_60m.jp2   ← Coastal aerosol
                      *_B09_60m.jp2   ← Water vapour

The bands available depend on the resolution folder.  This script loads
the 10 m and 20 m bands, resamples them to a common resolution, stacks
them into a cube, and applies the same normalisation used for the
benchmark datasets so the 3D CNN can be applied directly.

Dependencies
-------------
  pip install rasterio numpy scikit-image

Outputs
--------
  sentinel2_cube.npy          — (H, W, num_bands) float32 normalised cube
  outputs/visualizations/sentinel2_rgb_preview.png
"""

import os
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ_DIR = "outputs/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ─── Sentinel-2 band catalogue ────────────────────────────────────────────────
# (band_id, resolution_folder, description, central_wavelength_nm)
S2_BANDS = [
    ("B02", "R10m",  "Blue",           490),
    ("B03", "R10m",  "Green",          560),
    ("B04", "R10m",  "Red",            665),
    ("B08", "R10m",  "NIR",            842),
    ("B05", "R20m",  "Red-Edge-1",     705),
    ("B06", "R20m",  "Red-Edge-2",     740),
    ("B07", "R20m",  "Red-Edge-3",     783),
    ("B8A", "R20m",  "NIR-narrow",     865),
    ("B11", "R20m",  "SWIR-1",        1610),
    ("B12", "R20m",  "SWIR-2",        2190),
]


def find_safe_folder(base_dir: str = ".") -> str | None:
    """Find the first .SAFE folder inside base_dir."""
    matches = glob.glob(os.path.join(base_dir, "*.SAFE"))
    if not matches:
        matches = glob.glob(os.path.join(base_dir, "**", "*.SAFE"),
                            recursive=True)
    return matches[0] if matches else None


def _find_band_file(safe_path: str, band_id: str, res_folder: str) -> str | None:
    """
    Search inside a .SAFE folder for the .jp2 file for a given band.
    Returns the path or None if not found.
    """
    # pattern = os.path.join(
    #     safe_path, "GRANULE", "L2A_T43QCC_A008369_20260413T053935", "IMG_DATA",
    #     res_folder, f"*_{band_id}_{res_folder[1:]}*.jp2"
    # )
    pattern = os.path.join(
    safe_path,
    res_folder,
    f"*_{band_id}_{res_folder[1:]}*.jp2"
)
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        # try without resolution suffix in filename (some products omit it)
        # pattern2 = os.path.join(
        #     safe_path, "GRANULE", "L2A_T43QCC_A008369_20260413T053935", "IMG_DATA",
        #     res_folder, f"*_{band_id}_*.jp2"
        # )
        pattern2 = os.path.join(
    safe_path,
    res_folder,
    f"*_{band_id}_*.jp2"
)
        hits = glob.glob(pattern2, recursive=True)
    return hits[0] if hits else None


def load_sentinel2(
    safe_path: str | None = None,
    base_dir: str = ".",
    target_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load selected Sentinel-2 bands from a .SAFE folder and return a stacked
    hyperspectral-style cube.

    Parameters
    ----------
    safe_path    : path to the .SAFE folder (auto-detected if None)
    base_dir     : directory to search for .SAFE (used when safe_path=None)
    target_shape : (H, W) to resample all bands to (default: largest found size)

    Returns
    -------
    cube      : (H, W, num_bands)  float32  normalised to [0, 1]
    band_meta : list of dicts with 'band_id', 'description', 'wavelength_nm'
    """
    try:
        import rasterio
        from skimage.transform import resize as sk_resize
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Install with:  pip install rasterio scikit-image"
        )

    if safe_path is None:
        safe_path = find_safe_folder(base_dir)
        if safe_path is None:
            raise FileNotFoundError(
                f"No .SAFE folder found in '{base_dir}'.\n"
                "Set safe_path= to the full path of your .SAFE directory."
            )

    print(f"[Sentinel-2] Loading from: {safe_path}")

    arrays, band_meta, shapes = [], [], []

    for band_id, res_folder, description, wavelength in S2_BANDS:
        fpath = _find_band_file(safe_path, band_id, res_folder)
        if fpath is None:
            print(f"  [SKIP] {band_id} ({description}) — file not found in {res_folder}")
            continue

        with rasterio.open(fpath) as src:
            arr = src.read(1).astype(np.float32)   # (H, W)

        arrays.append(arr)
        band_meta.append({
            "band_id":      band_id,
            "description":  description,
            "wavelength_nm": wavelength,
            "source_file":  os.path.basename(fpath),
            "original_shape": arr.shape,
        })
        shapes.append(arr.shape)
        print(f"  [OK] {band_id:4s} ({description:15s})  shape={arr.shape}")

    if not arrays:
        raise RuntimeError("No bands could be loaded. Check your .SAFE folder structure.")

    # ── determine target shape ──────────────────────────────────────────────
    if target_shape is None:
        # use the shape of the largest (highest-resolution) band
        # target_shape = max(shapes, key=lambda s: s[0] * s[1])
        target_shape = (512, 512)

    print(f"\n[Sentinel-2] Resampling all bands to {target_shape}")

    # ── resample and stack ──────────────────────────────────────────────────
    resampled = []
    for arr, meta in zip(arrays, band_meta):
        if arr.shape != target_shape:
            arr = sk_resize(
                arr, target_shape,
                order=1,           # bilinear
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
        resampled.append(arr)

    cube = np.stack(resampled, axis=-1)   # (H, W, num_bands)

    # ── normalise to [0, 1] band-by-band ────────────────────────────────────
    # Sentinel-2 L2A reflectance values are scaled by 10000;
    # clip at 10000 then divide (values > 1 after div are set to 1)
    cube = np.clip(cube, 0, 10000) / 10000.0

    print(f"\n[Sentinel-2] Cube shape : {cube.shape}  (H, W, {len(band_meta)} bands)")
    print(f"             Value range: [{cube.min():.4f}, {cube.max():.4f}]")

    # ── save ─────────────────────────────────────────────────────────────────
    np.save("outputs/sentinel2_cube.npy", cube)
    print("[saved] outputs/sentinel2_cube.npy")

    # ── preview ──────────────────────────────────────────────────────────────
    _preview_rgb(cube, band_meta)

    return cube, band_meta


def _preview_rgb(cube: np.ndarray, band_meta: list[dict]) -> None:
    """Save a false-colour RGB preview using Red, Green, Blue bands."""
    ids    = [m["band_id"] for m in band_meta]
    r_idx  = ids.index("B04") if "B04" in ids else 0
    g_idx  = ids.index("B03") if "B03" in ids else 1
    b_idx  = ids.index("B02") if "B02" in ids else 2

    def _norm(a):
        mn, mx = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - mn) / (mx - mn + 1e-8), 0, 1)

    rgb = np.stack([
        _norm(cube[:, :, r_idx]),
        _norm(cube[:, :, g_idx]),
        _norm(cube[:, :, b_idx]),
    ], axis=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title("Sentinel-2 — True Colour Preview (B4/B3/B2)", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "sentinel2_rgb_preview.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def print_band_info(band_meta: list[dict]) -> None:
    """Print a formatted table of loaded bands."""
    print(f"\n{'Band':>6}  {'Description':20}  {'λ (nm)':>8}  {'Shape':>12}")
    print("-" * 55)
    for m in band_meta:
        print(f"{m['band_id']:>6}  {m['description']:20}  "
              f"{m['wavelength_nm']:>8}  {str(m['original_shape']):>12}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option 1: auto-detect .SAFE in current directory ──
    # cube, meta = load_sentinel2(base_dir=".")

    # ── Option 2: specify path explicitly ──
    SAFE_PATH = "data/S2C_MSIL2A_20260413T053021_N0512_R105_T43QCC_20260413T084816.SAFE/GRANULE/L2A_T43QCC_A008369_20260413T053935/IMG_DATA"
    try:
        cube, meta = load_sentinel2(safe_path=SAFE_PATH)
        print_band_info(meta)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please set SAFE_PATH to your actual .SAFE folder path.")
