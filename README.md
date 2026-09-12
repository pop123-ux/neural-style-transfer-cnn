# Neural Style Transfer CNN

[![CI](https://github.com/pop123-ux/neural-style-transfer-cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/neural-style-transfer-cnn/actions/workflows/ci.yml)

A neural style transfer project built with PyTorch, TorchVision, and a pretrained VGG19 convolutional network. It combines the content structure of one image with the visual statistics of another by directly optimizing a generated image against CNN feature losses.

## Features

- VGG19 feature extraction with frozen pretrained CNN weights
- content loss from higher-level convolutional activations
- style loss through Gram matrices across multiple VGG layers
- Adam optimization directly over generated image pixels
- local-file and URL image loading
- CLI for content/style transfer
- optional intermediate checkpoint images
- unit tests for tensor, loss, and image utilities
- Ruff linting and pytest validation through GitHub Actions

## Quick start

```bash
git clone https://github.com/pop123-ux/neural-style-transfer-cnn.git
cd neural-style-transfer-cnn
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run style transfer:

```bash
style-transfer --content examples/content-sample.png --style examples/style-sample.png -o outputs/stylized.png --steps 500
```

You can also use the backward-compatible script entry point:

```bash
python style_transfer.py --content path/to/content.jpg --style path/to/style.jpg -o outputs/stylized.png
```

## How it works

Neural style transfer uses a pretrained convolutional neural network as a **feature extractor**, not as a classifier.

The generated image starts as a copy of the content image. During optimization, the project updates those image pixels so that:

- its `conv4_2` representation stays close to the content image;
- its style-layer Gram matrices approach those of the style image.

The result preserves much of the content image's spatial structure while borrowing texture, color relationships, and feature correlations from the style image.

## Pipeline

```text
content image + style image
        ↓
resize and normalize with ImageNet statistics
        ↓
extract VGG19 feature maps
        ↓
content activations + style Gram matrices
        ↓
optimize generated pixels with Adam
        ↓
save the stylized image
```

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
├── examples/
├── src/
│   └── styletransfer/
│       ├── cli.py
│       ├── engine.py
│       ├── features.py
│       ├── image_io.py
│       └── models.py
├── tests/
│   ├── test_engine.py
│   ├── test_features.py
│   └── test_image_io.py
├── pyproject.toml
├── style_transfer.py
└── README.md
```

## Development

Install the editable package with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the same checks used by CI:

```bash
ruff check .
python -m pytest -q
```

## Continuous integration

The repository uses GitHub Actions through [`.github/workflows/ci.yml`](.github/workflows/ci.yml). The workflow runs automatically on pushes and pull requests targeting `main`, and can also be started manually from the Actions tab.

Each run creates a clean Ubuntu runner, installs Python 3.11, installs **CPU-only PyTorch/TorchVision**, installs the project with its development dependencies, and then executes:

```bash
ruff check .
python -m pytest -q
```

The CPU wheel is intentional: the CI suite validates source quality and deterministic unit behavior and does not run a full neural-style-transfer optimization job. Avoiding CUDA packages makes the workflow substantially smaller and faster while testing the same code paths relevant to these checks.

The badge at the top of this README reflects the latest workflow result.

## Notes

The first actual style-transfer run may download pretrained VGG19 weights through TorchVision. A GPU is recommended for larger images or many optimization steps, although CPU execution is supported.

The automated tests deliberately avoid downloading VGG19 weights or performing a long optimization run; they focus on the mathematical and utility components that should remain fast and deterministic in CI.

## License

MIT

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
