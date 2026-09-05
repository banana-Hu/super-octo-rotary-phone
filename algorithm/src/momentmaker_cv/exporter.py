"""Artifact export helpers."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from PIL import Image

from .contracts import PersonCutout, ProcessingResult, SubjectCutout
from .mask_processing import ProcessedMask
from .subject_processing import ProcessedSubjectMask

PERSON_ARTIFACT_PATTERN = re.compile(r"person_[0-9]{2,}\.png")
SUBJECT_ARTIFACT_PATTERN = re.compile(r"subject_[0-9]{2,}\.png")


def _temporary_sibling(target: Path) -> Path:
    return target.with_name(f".{target.stem}-{uuid.uuid4().hex}.tmp{target.suffix}")


def save_png_atomic(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def export_people(
    image: Image.Image,
    masks: list[ProcessedMask],
    output_dir: Path,
) -> tuple[tuple[PersonCutout, ...], list[Image.Image]]:
    people_dir = output_dir / "people"
    metadata: list[PersonCutout] = []
    cutout_images: list[Image.Image] = []

    for index, processed in enumerate(masks, start=1):
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(processed.alpha, mode="L"))
        cropped = rgba.crop(processed.crop_box)
        target = people_dir / f"person_{index:02d}.png"
        save_png_atomic(cropped, target)
        cutout_images.append(cropped)
        metadata.append(
            PersonCutout(
                person_id=index,
                confidence=round(processed.score, 6),
                source_box=processed.source_box,
                output_path=target,
                pixel_area=processed.pixel_area,
            )
        )

    return tuple(metadata), cutout_images


def export_subjects(
    image: Image.Image,
    masks: list[ProcessedSubjectMask],
    output_dir: Path,
) -> tuple[SubjectCutout, ...]:
    subjects_dir = output_dir / "subjects"
    metadata: list[SubjectCutout] = []
    for index, processed in enumerate(masks, start=1):
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(processed.alpha, mode="L"))
        cropped = rgba.crop(processed.crop_box)
        target = subjects_dir / f"subject_{index:02d}.png"
        save_png_atomic(cropped, target)
        metadata.append(
            SubjectCutout(
                subject_id=index,
                member_person_ids=processed.member_person_ids,
                mode=processed.mode,
                source_box=processed.source_box,
                output_path=target,
                pixel_area=processed.pixel_area,
            )
        )
    return tuple(metadata)


def cleanup_stale_people(
    output_dir: Path,
    current_people: tuple[PersonCutout, ...],
) -> tuple[str, ...]:
    """Remove only obsolete files that exactly match this module's naming scheme."""

    people_dir = output_dir / "people"
    if not people_dir.is_dir():
        return ()

    current_names = {person.output_path.name for person in current_people}
    warnings: list[str] = []
    for candidate in people_dir.iterdir():
        if not PERSON_ARTIFACT_PATTERN.fullmatch(candidate.name):
            continue
        if candidate.name in current_names:
            continue
        try:
            candidate.unlink()
        except OSError as exc:
            warnings.append(f"Could not remove stale artifact people/{candidate.name}: {exc}")
    return tuple(warnings)


def cleanup_stale_subjects(
    output_dir: Path,
    current_subjects: tuple[SubjectCutout, ...],
) -> tuple[str, ...]:
    subjects_dir = output_dir / "subjects"
    if not subjects_dir.is_dir():
        return ()
    current_names = {subject.output_path.name for subject in current_subjects}
    warnings: list[str] = []
    for candidate in subjects_dir.iterdir():
        if not SUBJECT_ARTIFACT_PATTERN.fullmatch(candidate.name):
            continue
        if candidate.name in current_names:
            continue
        try:
            candidate.unlink()
        except OSError as exc:
            warnings.append(f"Could not remove stale artifact subjects/{candidate.name}: {exc}")
    return tuple(warnings)


def export_manifest(result: ProcessingResult, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    try:
        temporary.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
