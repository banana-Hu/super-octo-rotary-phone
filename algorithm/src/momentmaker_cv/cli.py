"""Command-line entry point for local demos and backend integration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .contracts import CutoutOptions, ProcessingStatus
from .pipeline import process_image
from .segmenter import TorchvisionMaskRCNNSegmenter

EXIT_CODES = {
    ProcessingStatus.SUCCESS: 0,
    ProcessingStatus.NO_PERSON: 3,
    ProcessingStatus.INVALID_INPUT: 2,
    ProcessingStatus.MODEL_ERROR: 4,
    ProcessingStatus.PARTIAL_SUCCESS: 5,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momentmaker-cutout",
        description="Extract up to five people from an image as transparent PNG files.",
    )
    parser.add_argument("input", type=Path, help="JPG, PNG or WebP input image")
    parser.add_argument("--output", "-o", type=Path, required=True, help="output directory")
    parser.add_argument("--max-people", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = CutoutOptions(
            confidence_threshold=args.confidence,
            mask_threshold=args.mask_threshold,
            max_people=args.max_people,
            max_input_side=args.max_side,
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = process_image(
        args.input,
        args.output,
        options=options,
        segmenter=TorchvisionMaskRCNNSegmenter(device=args.device),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
