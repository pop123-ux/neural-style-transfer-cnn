# Neural Style Transfer CNN

[![CI](https://github.com/pop123-ux/neural-style-transfer-cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/neural-style-transfer-cnn/actions/workflows/ci.yml)

A portfolio-ready neural style transfer project built with PyTorch, TorchVision, and a VGG19 convolutional neural network. It combines the content of one image with the visual style of another by optimizing a generated image against CNN feature losses.

## Features

- VGG19 feature extraction with frozen pretrained CNN weights
- Content loss from high-level convolutional activations
- Style loss through Gram matrices across multiple VGG layers
- Adam optimization over generated image pixels
- Local file and URL image loading
- CLI for content/style image transfer
- Optional intermediate checkpoint images
- Tests for core tensor/image utilities

## Quick Start

```bash
git clone https://github.com/pop123-ux/neural-style-transfer-cnn.git
cd neural-style-transfer-cnn
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Run style transfer:

```bash
style-transfer --content examples\content-sample.png --style examples\style-sample.png -o outputs\stylized.png --steps 500
```

You can also run the compatibility script:

```bash
python style_transfer.py --content path\to\content.jpg --style path\to\style.jpg -o outputs\stylized.png
```

## How It Works

Neural style transfer uses a pretrained convolutional neural network as a feature extractor, not as a classifier.

The generated image starts as a copy of the content image. During optimization, the project updates the generated image pixels so that:

- its `conv4_2` features stay close to the content image
- its style-layer Gram matrices match the style image

The result keeps the structure of the content image while borrowing texture, color patterns, and brush-like statistics from the style image.

## Pipeline

```text
content image + style image
-> resize and normalize with ImageNet statistics
-> extract VGG19 features
-> compute content features
-> compute style Gram matrices
-> optimize generated image with Adam
-> save final stylized image
```

## Project Structure

```text
.
|-- src/styletransfer/
|   |-- cli.py
|   |-- engine.py
|   |-- features.py
|   |-- image_io.py
|   `-- models.py
|-- tests/
|-- examples/
|-- docs/
`-- style_transfer.py
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Notes

The first run may download pretrained VGG19 weights from TorchVision. Use a GPU if available; CPU execution works but can be slow for larger images or many optimization steps.

## License

MIT
