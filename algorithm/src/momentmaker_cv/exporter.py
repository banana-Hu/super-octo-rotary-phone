"""Artifact export helpers."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from PIL import Image

from .contracts import PersonCutout, ProcessingResult
from .mask_processing import ProcessedMask


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
