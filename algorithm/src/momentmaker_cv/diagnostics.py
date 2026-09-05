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

from .contracts import RESULT_SCHEMA_VERSION, CutoutOptions, ProcessingResult, ProcessingStatus
from .foreground import ForegroundSegmenter, InSPyReNetForegroundSegmenter
from .pipeline import process_image
from .segmenter import PersonSegmenter, TorchvisionMaskRCNNSegmenter

CHECKED_PACKAGES = ("numpy", "Pillow", "torch", "torchvision", "transparent-background")
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
    foreground_segmenter: ForegroundSegmenter | None = None,
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
    if foreground_segmenter is not None:
        try:
            foreground_segmenter.predict(Image.new("RGB", SMOKE_IMAGE_SIZE, "white"))
        except Exception as exc:
            return {
                "status": "error",
                "device": _active_device(detector),
                "foreground": "error",
                "versions": _versions(),
                "smoke_image_size": list(SMOKE_IMAGE_SIZE),
                "timing_ms": round((perf_counter() - started) * 1000, 2),
                "error": f"foreground model: {exc}",
            }
    return {
        "status": "ready",
        "device": _active_device(detector),
        "foreground": "ready" if foreground_segmenter is not None else "not_checked",
        "versions": _versions(),
        "smoke_image_size": list(SMOKE_IMAGE_SIZE),
        "timing_ms": round((perf_counter() - started) * 1000, 2),
        "error": None,
    }


def _validate_rgba_artifact(path: Path, output_dir: Path) -> list[str]:
    relative_path = _display_path(path, output_dir)
    if not path.is_file():
        return [f"{relative_path} is missing"]
    try:
        with Image.open(path) as cutout:
            if cutout.mode != "RGBA":
                return [f"{relative_path} is not RGBA"]
            errors: list[str] = []
            alpha_min, alpha_max = cutout.getchannel("A").getextrema()
            if alpha_min == 255:
                errors.append(f"{relative_path} has no transparent pixels")
            if alpha_max == 0:
                errors.append(f"{relative_path} is fully transparent")
            return errors
    except (OSError, UnidentifiedImageError) as exc:
        return [f"{relative_path} cannot be read: {exc}"]


def validate_result_artifacts(
    result: ProcessingResult,
    *,
    require_subjects: bool = True,
    require_foreground: bool = False,
) -> list[str]:
    """Return human-readable failures for a real-image smoke result."""

    errors: list[str] = []
    if result.status is not ProcessingStatus.SUCCESS:
        errors.append(f"processing status is {result.status.value}: {result.error or 'no details'}")
    if not result.people:
        errors.append("no person cutouts were produced")
    if require_subjects and not result.subjects:
        errors.append("no subject cutouts were produced")
    if require_foreground and any(subject.mode != "foreground" for subject in result.subjects):
        errors.append("foreground enhancement was requested but not applied")

    subject_ids = {subject.subject_id for subject in result.subjects}
    if result.subjects and result.primary_subject_id is None:
        errors.append("primary_subject_id is missing")
    if result.primary_subject_id not in subject_ids and result.primary_subject_id is not None:
        errors.append("primary_subject_id does not reference an exported subject")

    if result.preview_path is None or not result.preview_path.is_file():
        errors.append("preview.png is missing")
    if result.manifest_path is None or not result.manifest_path.is_file():
        errors.append("result.json is missing")

    for person in result.people:
        errors.extend(_validate_rgba_artifact(person.output_path, result.output_dir))
    person_ids = {person.person_id for person in result.people}
    for subject in result.subjects:
        errors.extend(_validate_rgba_artifact(subject.output_path, result.output_dir))
        if not set(subject.member_person_ids).issubset(person_ids):
            errors.append(f"subject {subject.subject_id} references a missing person cutout")

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
            manifest_subjects = manifest.get("subjects")
            if not isinstance(manifest_subjects, list) or len(manifest_subjects) != len(
                result.subjects
            ):
                errors.append("result.json subject count does not match exported cutouts")
            if manifest.get("primary_subject_id") != result.primary_subject_id:
                errors.append("result.json primary_subject_id does not match the Python result")
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
    parser.add_argument(
        "--subject-mode",
        choices=("none", "people", "foreground"),
        default="people",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image is None and args.output is not None:
        parser.error("--output requires an image")
    if args.image is not None and args.output is None:
        parser.error("--output is required when an image is provided")

    detector = TorchvisionMaskRCNNSegmenter(device=args.device)
    foreground_segmenter = (
        InSPyReNetForegroundSegmenter(device=args.device or "cpu")
        if args.subject_mode == "foreground"
        else None
    )
    if args.image is None:
        report = check_model(
            segmenter=detector,
            foreground_segmenter=foreground_segmenter,
        )
    else:
        started = perf_counter()
        result = process_image(
            args.image,
            args.output,
            options=CutoutOptions(subject_mode=args.subject_mode),
            segmenter=detector,
            foreground_segmenter=foreground_segmenter,
        )
        errors = validate_result_artifacts(
            result,
            require_subjects=args.subject_mode != "none",
            require_foreground=args.subject_mode == "foreground",
        )
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
