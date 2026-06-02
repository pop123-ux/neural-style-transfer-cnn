from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ModelSetup:
    """Loads the CNN backbone used for neural style transfer."""

    def __init__(self, model_name: str = "vgg19", device: torch.device | str | None = None, pretrained: bool = True) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.load_model(model_name, pretrained=pretrained).to(self.device).eval()
        self.freeze_parameters()

    def load_model(self, model_name: str, pretrained: bool = True) -> nn.Sequential:
        if model_name != "vgg19":
            raise ValueError(f"Unsupported model: {model_name}")

        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        return models.vgg19(weights=weights).features

    def freeze_parameters(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def get_model(self) -> nn.Sequential:
        return self.model

    def get_device(self) -> torch.device:
        return self.device
