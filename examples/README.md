# Examples

Put your content and style images here, then run:

```bash
style-transfer --content examples/content-sample.png --style examples/style-sample.png -o outputs/stylized.png --steps 500
```

Good test pairs:

- a portrait as the content image and a painting as the style image
- a city photo as the content image and abstract art as the style image
- a landscape as the content image and a textured artwork as the style image

Use a smaller `--max-size`, such as `256`, for faster CPU experiments.
