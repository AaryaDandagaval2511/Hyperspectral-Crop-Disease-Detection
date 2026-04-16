"""
run_pipeline.py — Master Orchestrator
=======================================
Runs the complete end-to-end hyperspectral pipeline in one command.

Usage
-----
    python run_pipeline.py

Edit the CONFIG block below to:
  • change which dataset is used
  • adjust patch size, epochs, PCA components, etc.
  • enable or skip individual steps

Expected data layout
---------------------
    data/
        Indian_pines_corrected.mat
        Indian_pines_gt.mat
        Salinas_corrected.mat
        Salinas_gt.mat
        PaviaU.mat
        PaviaU_gt.mat
    # For Step 11-12 (optional):
    data/
        S2A_MSIL2A_<date>.SAFE/   ← your Sentinel-2 .SAFE folder

Install dependencies
---------------------
    pip install numpy scipy matplotlib torch torchvision
                scikit-learn scikit-image seaborn rasterio
    pip install shap          # optional — for Step 10 Method A
"""

import os
import torch

# ──────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════╗
# ║                  C O N F I G                        ║
# ╚══════════════════════════════════════════════════════╝
DATASET       = "IndianPines"   # "IndianPines" | "Salinas" | "PaviaU"
PATCH_SIZE    = 7               # 5 | 7 | 9 | 11
NORM_METHOD   = "standard"      # "standard" | "minmax"
BATCH_SIZE    = 64
NUM_EPOCHS    = 5              # set lower (e.g. 20) for a quick smoke-test
LEARNING_RATE = 1e-3
PATIENCE      = 15              # early-stopping patience

# PCA experiment: which component counts to compare
PCA_COUNTS = [10, 20, 30, 50, 75, 100]

# Cross-sensor: OA/AA/Kappa of your full-band model from Step 8
# (leave as 0.0 if running the pipeline from scratch; they'll be filled in)
FULLBAND_OA    = 0.0
FULLBAND_AA    = 0.0
FULLBAND_KAPPA = 0.0

SAFE_PATH = None   # e.g. "data/S2A_MSIL2A_20230601_T32TNT.SAFE"
                   # set to None to skip Steps 11 and 12

# ──────────────────────────────────────────────────────────────────────────────

os.makedirs("outputs",               exist_ok=True)
os.makedirs("outputs/checkpoints",   exist_ok=True)
os.makedirs("outputs/visualizations",exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Hyperspectral CNN Pipeline")
print(f"  Dataset : {DATASET}")
print(f"  Device  : {device}")
print(f"{'='*60}\n")


# ── STEP 1 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 1 — Load Data ".ljust(60, "─"))
from step01_load_data import load_dataset
X, y = load_dataset(DATASET)


# ── STEP 2 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 2 — Visualize ".ljust(60, "─"))
from step02_visualize import visualize_dataset
visualize_dataset(X, y, DATASET)


# ── STEP 3 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 3 — Preprocess ".ljust(60, "─"))
from step03_preprocess import preprocess
X_norm, y_remap, labeled_mask, label_map = preprocess(X, y, norm_method=NORM_METHOD)


# ── STEP 4 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 4 — Patch Extraction ".ljust(60, "─"))
from step04_patch_extraction import extract_patches
X_patches, y_labels = extract_patches(X_norm, y_remap, patch_size=PATCH_SIZE)


# ── STEP 5 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 5 — Train / Val / Test Split ".ljust(60, "─"))
from step05_split_dataset import split_dataset, make_dataloaders
X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X_patches, y_labels)
train_loader, val_loader, test_loader = make_dataloaders(
    X_tr, X_v, X_te, y_tr, y_v, y_te, batch_size=BATCH_SIZE
)


# ── STEP 6 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 6 — Model Architecture ".ljust(60, "─"))
from step06_model import HybridSpectralNet, print_model_summary
num_bands   = X_norm.shape[2]
num_classes = int(y_labels.max()) + 1
model = HybridSpectralNet(num_bands, num_classes, patch_size=PATCH_SIZE).to(device)
print_model_summary(model, num_bands, PATCH_SIZE, device)


# ── STEP 7 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 7 — Training ".ljust(60, "─"))
from step07_train import train as train_model
history = train_model(
    model, train_loader, val_loader,
    num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    patience=PATIENCE, device=device,
    checkpoint_name="best_model",
)


# ── STEP 8 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 8 — Evaluation ".ljust(60, "─"))
from step08_evaluate import load_best_model, evaluate
from step02_visualize import CLASS_NAMES

model = load_best_model(model, "best_model", device)
names_0idx = CLASS_NAMES.get(DATASET, [])[1:]   # drop "Background"
metrics    = evaluate(model, test_loader, device,
                      class_names=names_0idx, dataset_name=DATASET)

# Update full-band metrics for cross-sensor comparison
FULLBAND_OA    = metrics["OA"]
FULLBAND_AA    = metrics["AA"]
FULLBAND_KAPPA = metrics["Kappa"]


# ── STEP 9 ─────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 9 — PCA Band Reduction ".ljust(60, "─"))
from step09_band_reduction import plot_explained_variance, pca_experiment
plot_explained_variance(X_tr, DATASET)
pca_results = pca_experiment(
    X_tr, X_v, X_te, y_tr, y_v, y_te,
    pc_counts=PCA_COUNTS,
    patch_size=PATCH_SIZE,
    num_classes=num_classes,
    device=device,
    dataset_name=DATASET,
    num_epochs=40,
)


# ── STEP 10 ────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 10 — Explainability ".ljust(60, "─"))
from step10_explainability import (
    gradient_band_importance, shap_band_importance, print_top_bands
)
# reload best full-band model
model = HybridSpectralNet(num_bands, num_classes, patch_size=PATCH_SIZE).to(device)
model = load_best_model(model, "best_model", device)

grad_scores = gradient_band_importance(model, X_te, y_te, device,
                                       dataset_name=DATASET)
print_top_bands(grad_scores)

shap_scores = shap_band_importance(model, X_te, device, dataset_name=DATASET)
if shap_scores is not None:
    print_top_bands(shap_scores)


# ── STEP 11 ────────────────────────────────────────────────────────────────
if SAFE_PATH:
    print("\n" + "▶ STEP 11 — Sentinel-2 Loading ".ljust(60, "─"))
    from step11_sentinel2 import load_sentinel2, print_band_info
    s2_cube, s2_meta = load_sentinel2(safe_path=SAFE_PATH)
    print_band_info(s2_meta)
else:
    print("\n[SKIP] STEP 11 — Set SAFE_PATH to load Sentinel-2 data.")


# ── STEP 12 ────────────────────────────────────────────────────────────────
print("\n" + "▶ STEP 12 — Cross-Sensor Testing ".ljust(60, "─"))
from step12_cross_sensor import (
    cross_sensor_experiment, plot_comparison,
    write_cross_sensor_report, get_sentinel2_equivalent_bands,
)
cs_metrics = cross_sensor_experiment(
    X_tr, X_v, X_te, y_tr, y_v, y_te,
    dataset_name=DATASET,
    patch_size=PATCH_SIZE,
    num_classes=num_classes,
    device=device,
    num_epochs=40,
)

n_s2 = len(get_sentinel2_equivalent_bands(DATASET))
comparison = {
    f"Full Band ({num_bands})":  {"OA": FULLBAND_OA,   "AA": FULLBAND_AA,
                                   "Kappa": FULLBAND_KAPPA},
    f"S2-equiv ({n_s2} bands)":  cs_metrics,
}
plot_comparison(comparison, DATASET)
write_cross_sensor_report(comparison, DATASET)


# ── DONE ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Pipeline complete!  All outputs saved to: outputs/")
print("="*60)
