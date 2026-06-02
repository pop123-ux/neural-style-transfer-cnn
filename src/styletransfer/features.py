from __future__ import annotations

import torch
from torch import nn

STYLE_LAYERS = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "28": "conv5_1",
}

CONTENT_LAYER = {"21": "conv4_2"}
DEFAULT_LAYERS = STYLE_LAYERS | CONTENT_LAYER


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Compute a normalized Gram matrix for a feature tensor."""

    batch_size, channels, height, width = tensor.shape
    features = tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)


class FeatureExtractor:
    """Extract named activations from selected layers of a CNN."""

    def __init__(self, model: nn.Sequential, layers: dict[str, str] | None = None) -> None:
        self.model = model
        self.layers = layers or DEFAULT_LAYERS

    def get_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        activation = image

        for name, layer in self.model._modules.items():
            activation = layer(activation)
            if name in self.layers:
                features[self.layers[name]] = activation

        return features
