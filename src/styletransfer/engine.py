from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .features import FeatureExtractor, gram_matrix
from .image_io import image_to_tensor, load_image, save_tensor_image
from .models import ModelSetup


@dataclass(frozen=True)
class StyleTransferConfig:
    """Configuration for neural style transfer optimization."""

    steps: int = 500
    learning_rate: float = 0.003
    content_weight: float = 1.0
    style_weight: float = 1e6
    max_size: int = 400
    target_size: int | None = None
    save_every: int = 100
    style_weights: dict[str, float] = field(
        default_factory=lambda: {
            "conv1_1": 1.0,
            "conv2_1": 0.75,
            "conv3_1": 0.2,
            "conv4_1": 0.2,
            "conv5_1": 0.2,
        }
    )


@dataclass(frozen=True)
class StyleTransferResult:
    """Final generated tensor and loss history from optimization."""

    image: torch.Tensor
    losses: list[float]


class StyleTransferEngine:
    """Optimizes an image to combine content features with style Gram matrices."""

    def __init__(
        self,
        config: StyleTransferConfig | None = None,
        model_setup: ModelSetup | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        self.config = config or StyleTransferConfig()
        self.model_setup = model_setup or ModelSetup()
        self.device = self.model_setup.get_device()
        self.feature_extractor = feature_extractor or FeatureExtractor(self.model_setup.get_model())

    def run(
        self,
        content_path: str | Path,
        style_path: str | Path,
        output_path: str | Path,
        checkpoint_dir: str | Path | None = None,
    ) -> StyleTransferResult:
        content = image_to_tensor(
            load_image(content_path),
            max_size=self.config.max_size,
            target_size=self.config.target_size,
            device=self.device,
        )
        style = image_to_tensor(
            load_image(style_path),
            max_size=self.config.max_size,
            target_size=self.config.target_size,
            device=self.device,
        )

        result = self.optimise(content, style, checkpoint_dir=checkpoint_dir)
        save_tensor_image(result.image, output_path)
        return result

    def optimise(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        checkpoint_dir: str | Path | None = None,
    ) -> StyleTransferResult:
        with torch.no_grad():
            content_features = self.feature_extractor.get_features(content)
            style_features = self.feature_extractor.get_features(style)
            style_grams = {
                layer_name: gram_matrix(style_features[layer_name])
                for layer_name in self.config.style_weights
            }

        generated = content.clone().requires_grad_(True).to(self.device)
        optimizer = optim.Adam([generated], lr=self.config.learning_rate)
        losses: list[float] = []

        for step in tqdm(range(1, self.config.steps + 1), desc="Optimizing style transfer"):
            target_features = self.feature_extractor.get_features(generated)
            content_loss, style_loss = self.compute_losses(target_features, content_features, style_grams)
            total_loss = self.config.content_weight * content_loss + self.config.style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_value = float(total_loss.item())
            losses.append(loss_value)

            if checkpoint_dir and self.config.save_every > 0 and step % self.config.save_every == 0:
                save_tensor_image(generated, Path(checkpoint_dir) / f"step_{step}.png")

        return StyleTransferResult(image=generated.detach(), losses=losses)

    def compute_losses(
        self,
        target_features: dict[str, torch.Tensor],
        content_features: dict[str, torch.Tensor],
        style_grams: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

        style_loss = torch.tensor(0.0, device=self.device)
        for layer_name, layer_weight in self.config.style_weights.items():
            target_gram = gram_matrix(target_features[layer_name])
            layer_loss = torch.mean((target_gram - style_grams[layer_name]) ** 2)
            style_loss = style_loss + layer_weight * layer_loss

        return content_loss, style_loss
