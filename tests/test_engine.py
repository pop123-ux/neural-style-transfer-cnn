import torch

from styletransfer.engine import StyleTransferConfig, StyleTransferEngine


def test_default_style_weights_include_expected_layers() -> None:
    config = StyleTransferConfig()

    assert set(config.style_weights) == {"conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"}


def test_compute_losses_returns_scalars() -> None:
    config = StyleTransferConfig()
    engine = StyleTransferEngine.__new__(StyleTransferEngine)
    engine.config = config
    engine.device = torch.device("cpu")

    target_features = {
        "conv4_2": torch.ones((1, 1, 2, 2)),
        "conv1_1": torch.ones((1, 1, 2, 2)),
        "conv2_1": torch.ones((1, 1, 2, 2)),
        "conv3_1": torch.ones((1, 1, 2, 2)),
        "conv4_1": torch.ones((1, 1, 2, 2)),
        "conv5_1": torch.ones((1, 1, 2, 2)),
    }
    content_features = {"conv4_2": torch.zeros((1, 1, 2, 2))}
    style_grams = {layer: torch.zeros((1, 1)) for layer in config.style_weights}

    content_loss, style_loss = engine.compute_losses(target_features, content_features, style_grams)

    assert content_loss.ndim == 0
    assert style_loss.ndim == 0
    assert content_loss.item() > 0
    assert style_loss.item() > 0
