"""Validated image loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(ValueError):
    """Raised when an input cannot be safely treated as a supported image."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def _fit_within(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    longest = max(width, height)
    if longest <= max_side:
        return size
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_image(
    path: str | Path,
    max_side: int = 1920,
    max_pixels: int = 40_000_000,
) -> LoadedImage:
    """Load a JPEG/PNG/WebP, apply EXIF orientation and optionally downscale it."""

    source = Path(path)
    if not source.is_file():
        raise InvalidImageError(f"input file does not exist: {source}")
    if max_side < 256:
        raise ValueError("max_side must be at least 256")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")

    try:
        with Image.open(source) as opened:
            if opened.format not in SUPPORTED_FORMATS:
                raise InvalidImageError(f"unsupported image format: {opened.format or 'unknown'}")
            if opened.width * opened.height > max_pixels:
                raise InvalidImageError(
                    f"image has too many pixels: {opened.width * opened.height} "
                    f"(limit: {max_pixels})"
                )
            opened.load()
            oriented = ImageOps.exif_transpose(opened)
            original_size = oriented.size
            rgb = oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"cannot read image: {source}") from exc

    target_size = _fit_within(original_size, max_side)
    if target_size != original_size:
        rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)

    return LoadedImage(
        image=rgb,
        original_size=original_size,
        processed_size=rgb.size,
    )
