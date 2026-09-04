"""Mask filtering, cleanup, feathering and crop calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .contracts import CutoutOptions
from .segmenter import MaskPrediction


@dataclass(frozen=True, slots=True)
class ProcessedMask:
    score: float
    source_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    alpha: np.ndarray
    pixel_area: int


def _is_severely_clipped(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> bool:
    """Reject narrow fragments cut by a side or the top of the source image."""

    width, height = image_size
    left, top, right, _bottom = box
    edge_margin_x = max(2, round(width * 0.01))
    edge_margin_y = max(2, round(height * 0.01))
    touches_cutting_edge = (
        left <= edge_margin_x or right >= width - edge_margin_x or top <= edge_margin_y
    )
    person_width_ratio = (right - left) / width
    return touches_cutting_edge and person_width_ratio < 0.12


def _resize_probability_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape == (size[1], size[0]):
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def _clean_binary_mask(binary: np.ndarray) -> np.ndarray:
    """Remove isolated pixels and close narrow holes using Pillow filters."""

    mask_image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    mask_image = mask_image.filter(ImageFilter.MedianFilter(size=3))
    mask_image = mask_image.filter(ImageFilter.MaxFilter(size=3))
    mask_image = mask_image.filter(ImageFilter.MinFilter(size=3))
    return np.asarray(mask_image, dtype=np.uint8) > 0


def _hard_alpha(binary: np.ndarray, feather_radius: float) -> np.ndarray:
    alpha_image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    if feather_radius:
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return np.asarray(alpha_image, dtype=np.uint8)


def _soft_alpha(
    probability: np.ndarray,
    binary: np.ndarray,
    mask_threshold: float,
) -> np.ndarray:
    """Map model probabilities to a soft edge constrained near the clean mask."""

    low = max(0.0, mask_threshold - 0.20)
    high = min(1.0, mask_threshold + 0.20)
    if high <= low:
        return binary.astype(np.uint8) * 255

    normalized = np.clip((probability - low) / (high - low), 0.0, 1.0)
    smooth = normalized * normalized * (3.0 - 2.0 * normalized)

    support_image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    support = np.asarray(support_image.filter(ImageFilter.MaxFilter(size=5))) > 0
    smooth *= support
    return np.rint(smooth * 255).astype(np.uint8)


def _clamp_box(
    box: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, min(width - 1, int(np.floor(box[0]))))
    top = max(0, min(height - 1, int(np.floor(box[1]))))
    right = max(left + 1, min(width, int(np.ceil(box[2]))))
    bottom = max(top + 1, min(height, int(np.ceil(box[3]))))
    return left, top, right, bottom


def _padded_mask_box(binary: np.ndarray, padding_ratio: float) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(binary)
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    padding = round(max(right - left, bottom - top) * padding_ratio)
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(binary.shape[1], right + padding),
        min(binary.shape[0], bottom + padding),
    )


def process_predictions(
    image_size: tuple[int, int],
    predictions: list[MaskPrediction],
    options: CutoutOptions,
) -> list[ProcessedMask]:
    """Apply deterministic quality filters and return up to ``max_people`` masks."""

    width, height = image_size
    minimum_area = width * height * options.min_area_ratio
    processed: list[ProcessedMask] = []

    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        if prediction.score < options.confidence_threshold:
            continue
        source_box = _clamp_box(prediction.box, image_size)
        if options.reject_severely_clipped and _is_severely_clipped(source_box, image_size):
            continue
        probability = _resize_probability_mask(prediction.mask, image_size)
        binary = _clean_binary_mask(probability >= options.mask_threshold)
        pixel_area = int(binary.sum())
        if pixel_area < minimum_area:
            continue

        crop_box = _padded_mask_box(binary, options.crop_padding_ratio)
        if options.alpha_mode == "soft":
            alpha = _soft_alpha(probability, binary, options.mask_threshold)
        else:
            alpha = _hard_alpha(binary, options.feather_radius)
        processed.append(
            ProcessedMask(
                score=prediction.score,
                source_box=source_box,
                crop_box=crop_box,
                alpha=alpha,
                pixel_area=pixel_area,
            )
        )
        if len(processed) >= options.max_people:
            break

    return sorted(
        processed,
        key=lambda item: (
            (item.source_box[0] + item.source_box[2]) / 2,
            item.source_box[1],
        ),
    )
