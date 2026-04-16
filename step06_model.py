"""
STEP 6 — 3D CNN Model
======================
Architecture: HybridSpectralNet
  Combines 3D convolutions (for joint spectral-spatial feature learning)
  followed by 2D convolutions (for deeper spatial refinement), then a
  fully-connected classifier head.

Input shape (one sample):
    (1, Bands, patch_size, patch_size)
    ↑ single input channel — Conv3d treats the Bands axis as depth

Detailed forward-pass dimensions (example: Bands=200, patch_size=7):

    Input          : (B, 1,   200, 7, 7)
    ─── 3D Block 1 ───────────────────────────────────────────────────
    Conv3d(1→8,  k=(7,3,3), pad=(3,1,1)) → (B, 8,  200, 7, 7)
    BatchNorm3d + ReLU
    Conv3d(8→16, k=(5,3,3), pad=(2,1,1)) → (B, 16, 200, 7, 7)
    BatchNorm3d + ReLU
    Conv3d(16→32,k=(3,3,3), pad=(1,1,1)) → (B, 32, 200, 7, 7)
    BatchNorm3d + ReLU
    ─── Reshape for 2D conv ──────────────────────────────────────────
    reshape → (B, 32*200, 7, 7) = (B, 6400, 7, 7)
    Conv2d(6400→64, k=1)        → (B, 64,   7, 7)    # 1×1 conv = channel mixer
    BatchNorm2d + ReLU
    ─── 2D Block ─────────────────────────────────────────────────────
    Conv2d(64→128, k=3, pad=1)  → (B, 128, 7, 7)
    BatchNorm2d + ReLU
    AdaptiveAvgPool2d(1×1)      → (B, 128, 1, 1)
    ─── Classifier ───────────────────────────────────────────────────
    Flatten                     → (B, 128)
    Dropout(0.4)
    Linear(128 → num_classes)   → (B, num_classes)

Total trainable parameters: ~ 50 K–500 K depending on Bands and num_classes.

Note: The architecture is intentionally kept lightweight so it trains on a
      CPU in reasonable time.  For GPU training no code changes are needed —
      just move the model and data to CUDA as shown in step07_train.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSpectralNet(nn.Module):
    """
    Lightweight 3D-CNN for hyperspectral image classification.

    Parameters
    ----------
    num_bands   : number of spectral bands (depth dimension)
    num_classes : number of output classes (background already excluded)
    patch_size  : spatial size of the input patch (height = width)
    dropout     : dropout rate before the final linear layer
    """

    def __init__(
        self,
        num_bands: int,
        num_classes: int,
        patch_size: int = 7,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_bands   = num_bands
        self.num_classes = num_classes
        self.patch_size  = patch_size

        # ─── 3D convolutional blocks ─────────────────────────────────────────
        # kernel: (spectral_depth, height, width)
        self.conv3d_block = nn.Sequential(
            # block 1
            nn.Conv3d(1, 8,  kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            # block 2
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # block 3
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # After 3D blocks the shape is (B, 32, num_bands, P, P)
        # Reshape to (B, 32*num_bands, P, P) and use a 1×1 conv to compress

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(32 * num_bands, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ─── 2D convolutional block ──────────────────────────────────────────
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # global average pool → (B, 128, 1, 1)
        )

        # ─── Classifier head ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # ─── Weight initialisation ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, Bands, patch_size, patch_size)

        Returns
        -------
        logits : (batch, num_classes)
        """
        # ── 3D convolutions ──
        out = self.conv3d_block(x)           # (B, 32, Bands, P, P)

        # ── reshape: merge filters × spectral into one channel dim ──
        B, C, D, H, W = out.shape           # C=32, D=Bands
        out = out.view(B, C * D, H, W)      # (B, 32*Bands, P, P)

        # ── 1×1 conv channel mixer ──
        out = self.channel_mixer(out)        # (B, 64, P, P)

        # ── 2D convolutions + global pooling ──
        out = self.conv2d_block(out)         # (B, 128, 1, 1)

        # ── classification ──
        logits = self.classifier(out)        # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Helper: count parameters
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(
    model: nn.Module,
    num_bands: int,
    patch_size: int,
    device: torch.device,
) -> None:
    """
    Print architecture string and run one dummy forward pass to verify shapes.
    """
    print("\n" + "=" * 60)
    print(model)
    print("=" * 60)
    print(f"Trainable parameters: {count_parameters(model):,}")

    dummy = torch.zeros(1, 1, num_bands, patch_size, patch_size).to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    print("Model summary check passed ✓")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Indian Pines corrected: 200 bands, 16 classes, 7×7 patches
    model = HybridSpectralNet(
        num_bands=200, num_classes=16, patch_size=7
    ).to(device)

    print_model_summary(model, num_bands=200, patch_size=7, device=device)
