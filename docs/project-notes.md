# Project Notes

This project is based on a neural style transfer workflow using convolutional neural networks.

## Core Concepts

- Artificial intelligence simulates intelligent behavior such as learning, reasoning, and perception.
- Machine learning learns patterns from data instead of relying only on explicit rules.
- Deep learning uses neural networks with many layers to learn representations from data.
- Convolutional neural networks are especially useful for images because convolution layers learn spatial feature maps.

## Neural Style Transfer

Neural style transfer separates two visual ideas:

- Content: the objects and structure in the content image.
- Style: the texture, color, and repeated visual patterns in the style image.

The project uses VGG19 as a fixed feature extractor. The VGG model is not trained here. Its parameters are frozen, and the generated image itself is optimized.

## Losses

Content loss compares generated-image features with content-image features.

Style loss compares Gram matrices from generated-image features with Gram matrices from style-image features. A Gram matrix captures correlations between feature channels, which makes it useful for representing texture-like style.

The final objective combines both losses:

```text
total loss = content weight * content loss + style weight * style loss
```

Gradient descent then updates the generated image pixels to reduce that total loss.
