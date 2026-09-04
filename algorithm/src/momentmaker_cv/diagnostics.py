"""Local model and artifact checks for demo-machine preparation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from .contracts import RESULT_SCHEMA_VERSION, ProcessingResult, ProcessingStatus
from .pipeline import process_image
from .segmenter import PersonSegmenter, TorchvisionMaskRCNNSegmenter

CHECKED_PACKAGES = ("numpy", "Pillow", "torch", "torchvision")
SMOKE_IMAGE_SIZE = (64, 64)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in CHECKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _active_device(segmenter: PersonSegmenter) -> str | None:
    device = getattr(segmenter, "active_device", None)
    return str(device) if device is not None else None


def _display_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def check_model(
    device: str | None = None,
    segmenter: PersonSegmenter | None = None,
) -> dict[str, Any]:
    """Load the model and run one small inference to verify its runtime."""

    detector = segmenter or TorchvisionMaskRCNNSegmenter(device=device)
    started = perf_counter()
    try:
        detector.predict(Image.new("RGB", SMOKE_IMAGE_SIZE, "white"))
    except Exception as exc:
        return {
            "status": "error",
            "device": _active_device(detector),
            "versions": _versions(),
            "smoke_image_size": list(SMOKE_IMAGE_SIZE),
            "timing_ms": round((perf_counter() - started) * 1000, 2),
            "error": str(exc),
        }
    return {
        "status": "ready",
        "device": _active_device(detector),
        "versions": _versions(),
        "smoke_image_size": list(SMOKE_IMAGE_SIZE),
        "timing_ms": round((perf_counter() - started) * 1000, 2),
        "error": None,
    }


def validate_result_artifacts(result: ProcessingResult) -> list[str]:
    """Return human-readable failures for a real-image smoke result."""

    errors: list[str] = []
    if result.status is not ProcessingStatus.SUCCESS:
        errors.append(f"processing status is {result.status.value}: {result.error or 'no details'}")
    if not result.people:
        errors.append("no person cutouts were produced")

    if result.preview_path is None or not result.preview_path.is_file():
        errors.append("preview.png is missing")
    if result.manifest_path is None or not result.manifest_path.is_file():
        errors.append("result.json is missing")

    for person in result.people:
        relative_path = _display_path(person.output_path, result.output_dir)
        if not person.output_path.is_file():
            errors.append(f"{relative_path} is missing")
            continue
        try:
            with Image.open(person.output_path) as cutout:
                if cutout.mode != "RGBA":
                    errors.append(f"{relative_path} is not RGBA")
                    continue
                alpha_min, alpha_max = cutout.getchannel("A").getextrema()
                if alpha_min == 255:
                    errors.append(f"{relative_path} has no transparent pixels")
                if alpha_max == 0:
                    errors.append(f"{relative_path} is fully transparent")
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"{relative_path} cannot be read: {exc}")

    if result.manifest_path is not None and result.manifest_path.is_file():
        try:
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                errors.append("result.json must contain a JSON object")
                return errors
            if manifest.get("schema_version") != RESULT_SCHEMA_VERSION:
                errors.append("result.json has an unexpected schema version")
            if manifest.get("status") != result.status.value:
                errors.append("result.json status does not match the Python result")
            manifest_people = manifest.get("people")
            if not isinstance(manifest_people, list) or len(manifest_people) != len(result.people):
                errors.append("result.json person count does not match exported cutouts")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"result.json cannot be read: {exc}")

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momentmaker-check",
        description="Prepare the local model or verify a real-image cutout run.",
    )
    parser.add_argument("image", nargs="?", type=Path, help="optional real image for full QA")
    parser.add_argument("--output", "-o", type=Path, help="output directory for real-image QA")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image is None and args.output is not None:
        parser.error("--output requires an image")
    if args.image is not None and args.output is None:
        parser.error("--output is required when an image is provided")

    detector = TorchvisionMaskRCNNSegmenter(device=args.device)
    if args.image is None:
        report = check_model(segmenter=detector)
    else:
        started = perf_counter()
        result = process_image(args.image, args.output, segmenter=detector)
        errors = validate_result_artifacts(result)
        report = {
            "status": "ready" if not errors else "error",
            "device": _active_device(detector),
            "versions": _versions(),
            "timing_ms": round((perf_counter() - started) * 1000, 2),
            "errors": errors,
            "result": result.to_dict(),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
