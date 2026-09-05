"""Build template-ready subjects from person masks and optional foreground saliency."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

from .mask_processing import ProcessedMask


@dataclass(frozen=True, slots=True)
class ProcessedSubjectMask:
    member_person_ids: tuple[int, ...]
    source_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    alpha: np.ndarray
    pixel_area: int
    mode: Literal["people", "foreground"]


def _vertical_overlap_ratio(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    minimum_height = min(first[3] - first[1], second[3] - second[1])
    return overlap / minimum_height if minimum_height else 0.0


def _people_are_near(
    first: ProcessedMask,
    second: ProcessedMask,
    horizontal_gap_ratio: float,
    minimum_vertical_overlap: float,
    minimum_area_similarity: float,
) -> bool:
    first_width = first.source_box[2] - first.source_box[0]
    second_width = second.source_box[2] - second.source_box[0]
    horizontal_gap = max(
        0,
        max(first.source_box[0], second.source_box[0])
        - min(first.source_box[2], second.source_box[2]),
    )
    area_similarity = min(first.pixel_area, second.pixel_area) / max(
        first.pixel_area, second.pixel_area
    )
    return (
        area_similarity >= minimum_area_similarity
        and horizontal_gap <= min(first_width, second_width) * horizontal_gap_ratio
        and _vertical_overlap_ratio(first.source_box, second.source_box) >= minimum_vertical_overlap
    )


def group_nearby_people(
    people: list[ProcessedMask],
    *,
    horizontal_gap_ratio: float = 0.35,
    minimum_vertical_overlap: float = 0.25,
    minimum_area_similarity: float = 0.20,
) -> list[tuple[int, ...]]:
    """Return transitive, deterministic groups using one-based person IDs."""

    remaining = set(range(len(people)))
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        members = {seed}
        pending = [seed]
        while pending:
            current = pending.pop()
            neighbors = {
                candidate
                for candidate in remaining
                if _people_are_near(
                    people[current],
                    people[candidate],
                    horizontal_gap_ratio,
                    minimum_vertical_overlap,
                    minimum_area_similarity,
                )
            }
            remaining.difference_update(neighbors)
            members.update(neighbors)
            pending.extend(neighbors)
        groups.append(tuple(index + 1 for index in sorted(members)))
    return groups


def _mask_box(binary: np.ndarray, padding_ratio: float = 0.04) -> tuple[int, int, int, int]:
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


def _expanded_group_box(
    group: tuple[int, ...], people: list[ProcessedMask], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    boxes = [people[person_id - 1].source_box for person_id in group]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    padding = round(max(right - left, bottom - top) * 0.50)
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image_size[0], right + padding),
        min(image_size[1], bottom + padding),
    )


def _connected_to_people(candidate: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Select candidate components touching a dilated person anchor at bounded resolution."""

    height, width = candidate.shape
    scale = min(1.0, 512 / max(width, height))
    reduced_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    candidate_small = (
        np.asarray(
            Image.fromarray(candidate.astype(np.uint8) * 255).resize(
                reduced_size, Image.Resampling.NEAREST
            )
        )
        > 0
    )
    anchor_image = Image.fromarray(anchor.astype(np.uint8) * 255).resize(
        reduced_size, Image.Resampling.NEAREST
    )
    anchor_small = np.asarray(anchor_image.filter(ImageFilter.MaxFilter(size=7))) > 0
    seeds = np.argwhere(candidate_small & anchor_small)
    reachable = np.zeros_like(candidate_small)
    queue: deque[tuple[int, int]] = deque()
    for row, column in seeds:
        point = (int(row), int(column))
        if not reachable[point]:
            reachable[point] = True
            queue.append(point)

    small_height, small_width = candidate_small.shape
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if not (0 <= next_row < small_height and 0 <= next_column < small_width):
                continue
            if candidate_small[next_row, next_column] and not reachable[next_row, next_column]:
                reachable[next_row, next_column] = True
                queue.append((next_row, next_column))

    restored = Image.fromarray(reachable.astype(np.uint8) * 255).resize(
        (width, height), Image.Resampling.NEAREST
    )
    return (np.asarray(restored) > 0) & candidate


def build_subject_masks(
    image_size: tuple[int, int],
    people: list[ProcessedMask],
    foreground: np.ndarray | None = None,
) -> list[ProcessedSubjectMask]:
    """Combine nearby people and foreground connected to each group."""

    expected_shape = (image_size[1], image_size[0])
    if foreground is not None and foreground.shape != expected_shape:
        raise ValueError(f"foreground mask shape must be {expected_shape}, got {foreground.shape}")
    probability = None
    if foreground is not None:
        probability = np.asarray(foreground, dtype=np.float32)
        if probability.size and float(probability.max()) > 1.0:
            probability = probability / 255.0
        probability = np.clip(probability, 0.0, 1.0)

    subjects: list[ProcessedSubjectMask] = []
    for group in group_nearby_people(people):
        person_alpha = np.maximum.reduce([people[index - 1].alpha for index in group])
        alpha = person_alpha.copy()
        mode: Literal["people", "foreground"] = "people"
        if probability is not None:
            left, top, right, bottom = _expanded_group_box(group, people, image_size)
            candidate = probability[top:bottom, left:right] >= 0.15
            anchor = person_alpha[top:bottom, left:right] > 0
            other_ids = set(range(1, len(people) + 1)).difference(group)
            if other_ids:
                other_alpha = np.maximum.reduce([people[index - 1].alpha for index in other_ids])
                other_region = Image.fromarray(
                    (other_alpha[top:bottom, left:right] > 0).astype(np.uint8) * 255
                ).filter(ImageFilter.MaxFilter(size=5))
                candidate &= np.asarray(other_region) == 0
            connected = _connected_to_people(candidate, anchor)
            enhanced = np.zeros_like(person_alpha)
            enhanced_region = np.rint(probability[top:bottom, left:right] * 255).astype(np.uint8)
            enhanced[top:bottom, left:right] = enhanced_region * connected.astype(np.uint8)
            alpha = np.maximum(alpha, enhanced)
            mode = "foreground"

        binary = alpha > 0
        crop_box = _mask_box(binary)
        subjects.append(
            ProcessedSubjectMask(
                member_person_ids=group,
                source_box=_mask_box(binary, padding_ratio=0),
                crop_box=crop_box,
                alpha=alpha,
                pixel_area=int(binary.sum()),
                mode=mode,
            )
        )
    return subjects
