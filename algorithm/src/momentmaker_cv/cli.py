"""Command-line entry point for local demos and backend integration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .contracts import CutoutOptions, ProcessingStatus
from .foreground import InSPyReNetForegroundSegmenter
from .pipeline import process_image
from .segmenter import TorchvisionMaskRCNNSegmenter

EXIT_CODES = {
    ProcessingStatus.SUCCESS: 0,
    ProcessingStatus.NO_PERSON: 3,
    ProcessingStatus.INVALID_INPUT: 2,
    ProcessingStatus.MODEL_ERROR: 4,
    ProcessingStatus.PARTIAL_SUCCESS: 5,
    ProcessingStatus.PROCESSING_ERROR: 6,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momentmaker-cutout",
        description="Extract people and template-ready subject groups as transparent PNG files.",
    )
    parser.add_argument("input", type=Path, help="JPG, PNG or WebP input image")
    parser.add_argument("--output", "-o", type=Path, required=True, help="output directory")
    parser.add_argument("--max-people", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--alpha-mode", choices=("hard", "soft"), default="soft")
    parser.add_argument(
        "--subject-mode",
        choices=("none", "people", "foreground"),
        default="people",
        help=(
            "none: people only; people: add nearby-person groups; "
            "foreground: retain connected objects"
        ),
    )
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--keep-partial-people",
        action="store_true",
        help="keep narrow person fragments clipped by the left, right or top edge",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = CutoutOptions(
            confidence_threshold=args.confidence,
            mask_threshold=args.mask_threshold,
            alpha_mode=args.alpha_mode,
            max_people=args.max_people,
            max_input_side=args.max_side,
            max_input_pixels=args.max_pixels,
            reject_severely_clipped=not args.keep_partial_people,
            subject_mode=args.subject_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    foreground_segmenter = (
        InSPyReNetForegroundSegmenter(device=args.device or "cpu")
        if args.subject_mode == "foreground"
        else None
    )
    result = process_image(
        args.input,
        args.output,
        options=options,
        segmenter=TorchvisionMaskRCNNSegmenter(device=args.device),
        foreground_segmenter=foreground_segmenter,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
