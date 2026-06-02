from __future__ import annotations

import argparse
from pathlib import Path

from .engine import StyleTransferConfig, StyleTransferEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="style-transfer",
        description="Run neural style transfer with a VGG19 convolutional feature extractor.",
    )
    parser.add_argument("--content", required=True, help="Path or URL for the content image.")
    parser.add_argument("--style", required=True, help="Path or URL for the style image.")
    parser.add_argument("-o", "--output", default="outputs/stylized.png", help="Output image path.")
    parser.add_argument("--steps", type=int, default=500, help="Number of optimization steps.")
    parser.add_argument("--max-size", type=int, default=400, help="Maximum image size before optimization.")
    parser.add_argument("--target-size", type=int, default=None, help="Optional square crop size.")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Content loss weight.")
    parser.add_argument("--style-weight", type=float, default=1e6, help="Style loss weight.")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Adam optimizer learning rate.")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint interval in steps.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional directory for intermediate images.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StyleTransferConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        max_size=args.max_size,
        target_size=args.target_size,
        save_every=args.save_every,
    )
    result = StyleTransferEngine(config=config).run(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"Saved stylized image to {args.output}")
    print(f"Final loss: {result.losses[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
