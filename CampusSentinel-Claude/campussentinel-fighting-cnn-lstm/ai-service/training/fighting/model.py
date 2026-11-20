"""CNN-LSTM fight classifier.

    input  (B, T, 3, 224, 224)  ImageNet-normalised float frames (T = 16)
      │ reshape → (B·T, 3, 224, 224)
    ResNet18 (ImageNet-pretrained, final fc removed, frozen by default)
      │ → (B·T, 512)  one feature vector per frame
      │ reshape → (B, T, 512)
    LSTM (batch_first) over the 16 frame features
      │ final hidden state of the last layer → (B, hidden)
    Head: Dropout → Linear → ReLU → Dropout → Linear
      ▼
    output (B, 1) raw LOGIT. P(Fight) = sigmoid(logit). No sigmoid inside the model.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from training.fighting.config import ModelConfig


class CNNLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        resnet = models.resnet18(weights=weights)
        # Everything except the final fc layer. avgpool gives (N, 512, 1, 1).
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                  resnet.layer1, resnet.layer2, resnet.layer3)
        self.layer4 = resnet.layer4
        self.pool = resnet.avgpool

        self.lstm = nn.LSTM(
            input_size=cfg.feature_dim, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
            batch_first=True, dropout=cfg.lstm_dropout if cfg.num_layers > 1 else 0.0,
        )
        mid = max(cfg.hidden_size // 2, 16)
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.hidden_size, mid),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(mid, 1),
        )
        self.set_backbone_trainability(cfg.freeze_backbone, cfg.train_layer4)

    # ------------------------------------------------------------------ backbone control
    def set_backbone_trainability(self, freeze: bool, train_layer4: bool = False) -> None:
        """freeze=True: ResNet18 frozen (default). train_layer4=True additionally unfreezes layer4."""
        self.cfg.freeze_backbone, self.cfg.train_layer4 = freeze, train_layer4
        for p in self.stem.parameters():
            p.requires_grad = not freeze
        for p in self.layer4.parameters():
            p.requires_grad = (not freeze) or train_layer4

    @property
    def fully_frozen(self) -> bool:
        return self.cfg.freeze_backbone and not self.cfg.train_layer4

    def train(self, mode: bool = True) -> "CNNLSTM":
        """Frozen ResNet parts stay in eval mode even while training, so their BatchNorm
        running statistics are not overwritten by our small batches."""
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.stem.eval()
            if not self.cfg.train_layer4:
                self.layer4.eval()
        return self

    def backbone_parameters(self):
        return [p for m in (self.stem, self.layer4) for p in m.parameters() if p.requires_grad]

    def temporal_parameters(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())

    # ------------------------------------------------------------------ forward
    def frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) → (N, 512). Frozen parts run without building a gradient graph."""
        if self.cfg.freeze_backbone:
            with torch.no_grad():
                x = self.stem(frames)
                if not self.cfg.train_layer4:
                    x = self.layer4(x)
            if self.cfg.train_layer4:
                x = self.layer4(x)
        else:
            x = self.layer4(self.stem(frames))
        return torch.flatten(self.pool(x), 1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(clips.shape)}")
        b, t = clips.shape[:2]
        # (B, T, C, H, W) → (B·T, C, H, W): all frames of all clips in ONE CNN call.
        feats = self.frame_features(clips.reshape(b * t, *clips.shape[2:]))
        # (B·T, 512) → (B, T, 512): row-major reshape restores [clip][frame] order exactly.
        feats = feats.reshape(b, t, -1)
        _, (h_n, _) = self.lstm(feats)        # h_n: (num_layers, B, hidden)
        return self.head(h_n[-1])             # last layer's final hidden state → (B, 1) logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
