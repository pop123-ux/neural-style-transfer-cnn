from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path_or_url: str | Path, timeout: int = 30) -> Image.Image:
    """Load a local or remote image as RGB."""

    source = str(path_or_url)
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, timeout=timeout)
        response.raise_for_status()
        stream: BinaryIO = BytesIO(response.content)
        return Image.open(stream).convert("RGB")
    return Image.open(source).convert("RGB")


def build_transform(max_size: int = 400, target_size: int | None = None) -> transforms.Compose:
    """Create the preprocessing pipeline expected by pretrained VGG models."""

    size = target_size or max_size
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def image_to_tensor(
    image: Image.Image,
    max_size: int = 400,
    target_size: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a PIL image into a normalized tensor batch."""

    transform = build_transform(max_size=max_size, target_size=target_size)
    return transform(image)[:3, :, :].unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor batch back to a PIL RGB image."""

    image = tensor.detach().cpu().clone().squeeze(0)
    for channel, mean, std in zip(image, IMAGENET_MEAN, IMAGENET_STD, strict=True):
        channel.mul_(std).add_(mean)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a normalized tensor batch as an image file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(tensor).save(output_path)
