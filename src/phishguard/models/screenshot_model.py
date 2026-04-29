"""EfficientNet-B0 classifier for phishing screenshots."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ScreenshotClassifier(nn.Module):  # type: ignore[misc]
    """EfficientNet-B0 backbone with a 2-class head and dropout."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
