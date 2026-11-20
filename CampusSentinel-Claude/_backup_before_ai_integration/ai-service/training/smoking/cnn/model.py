"""SmokingCNN — a small convolutional network trained FROM SCRATCH.

Task:   binary image classification, smoking (1) vs. not_smoking (0).
Input:  float tensor (N, 3, 224, 224), RGB, normalised during preprocessing.
Output: float tensor (N, 1) of raw LOGITS (any real number).

No sigmoid is applied in forward(). Training uses nn.BCEWithLogitsLoss, which
combines sigmoid + binary cross-entropy in one numerically stable operation.
To get a probability at inference time, call `predict_proba()` or apply
torch.sigmoid() yourself.

This file is the single source of truth for the architecture. Training
(Colab) and local inference both import it, so the saved weights always
match the code that loads them.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 224
INPUT_CHANNELS = 3
CLASS_NAMES = ("not_smoking", "smoking")  # index 1 = positive class


class ConvBlock(nn.Sequential):
    """Conv3x3 → BatchNorm → ReLU → MaxPool(2). Halves height and width.

    bias=False on the convolution: BatchNorm immediately subtracts the mean
    and adds its own learnable shift (beta), so a conv bias would be cancelled
    out — it is redundant parameters.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class SmokingCNN(nn.Module):
    """Four conv blocks (32 → 64 → 128 → 256) + small classification head.

    Spatial size through the network for a 224×224 input:
        224 → 112 → 56 → 28 → 14   (each MaxPool halves it)
    Final feature map: 256 channels × 14 × 14, averaged to a 256-dim vector.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        hidden_units: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if len(channels) == 0:
            raise ValueError("channels must contain at least one block")

        blocks = []
        prev = in_channels
        for out in channels:
            blocks.append(ConvBlock(prev, out))
            prev = out
        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (N, 256, H, W) → (N, 256, 1, 1), works for any H, W
            nn.Flatten(),              # → (N, 256)
            nn.Linear(prev, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),     # active only in model.train(); off in model.eval()
            nn.Linear(hidden_units, 1),  # one logit: > 0 leans "smoking", < 0 leans "not smoking"
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming (He) init suits ReLU networks: it keeps activation variance
        roughly stable from layer to layer, which helps training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        # Output layer: start with tiny weights so initial logits are ≈ 0,
        # i.e. the untrained model predicts ≈ 0.5 for every image. That gives an
        # initial loss of ≈ ln(2) ≈ 0.693 instead of large, confident random guesses.
        out_layer = self.classifier[-1]
        nn.init.normal_(out_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(out_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"expected input of shape (N, {INPUT_CHANNELS}, H, W), got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x))  # raw logits, shape (N, 1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probability of the 'smoking' class, shape (N,). Switches to eval mode."""
        self.eval()
        return torch.sigmoid(self.forward(x)).squeeze(1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
