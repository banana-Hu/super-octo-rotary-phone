"""Generate a neutral contact-sheet preview for demos and QA."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .exporter import save_png_atomic


def _contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def create_preview(cutouts: list[Image.Image], target: Path) -> None:
    canvas = Image.new("RGB", (1280, 720), "#EEF1F5")
    draw = ImageDraw.Draw(canvas)
    slots = [(55 + column * 244, 120, 275 + column * 244, 620) for column in range(5)]

    for index, slot in enumerate(slots):
        left, top, right, bottom = slot
        draw.rounded_rectangle(slot, radius=24, fill="#FFFFFF", outline="#D8DEE8", width=2)
        if index >= len(cutouts):
            continue
        fitted = _contain(cutouts[index], (right - left - 28, bottom - top - 28))
        x = left + (right - left - fitted.width) // 2
        y = bottom - 14 - fitted.height
        canvas.paste(fitted, (x, y), fitted)

    save_png_atomic(canvas, target)
