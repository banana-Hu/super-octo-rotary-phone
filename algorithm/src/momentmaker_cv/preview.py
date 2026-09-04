"""Generate a neutral contact-sheet preview for demos and QA."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .exporter import save_png_atomic

PREVIEW_LIGHT_BG = "#F8FAFC"
PREVIEW_DARK_BG = "#263241"


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
        card_size = (right - left, bottom - top)
        card = Image.new("RGB", card_size, PREVIEW_LIGHT_BG)
        ImageDraw.Draw(card).rectangle(
            (card_size[0] // 2, 0, card_size[0], card_size[1]),
            fill=PREVIEW_DARK_BG,
        )
        rounded_mask = Image.new("L", card_size, 0)
        ImageDraw.Draw(rounded_mask).rounded_rectangle(
            (0, 0, card_size[0] - 1, card_size[1] - 1),
            radius=24,
            fill=255,
        )
        canvas.paste(card, (left, top), rounded_mask)
        draw.rounded_rectangle(slot, radius=24, outline="#D8DEE8", width=2)
        if index >= len(cutouts):
            continue
        fitted = _contain(cutouts[index], (right - left - 28, bottom - top - 28))
        x = left + (right - left - fitted.width) // 2
        y = bottom - 14 - fitted.height
        canvas.paste(fitted, (x, y), fitted)

    save_png_atomic(canvas, target)
