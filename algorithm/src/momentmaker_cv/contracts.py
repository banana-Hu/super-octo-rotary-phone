"""Stable input and output contracts for the cutout pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

RESULT_SCHEMA_VERSION = "1.0"


def _portable_path(path: Path) -> str:
    return path.as_posix()


def _artifact_path(path: Path, output_dir: Path | None) -> str:
    if output_dir is None:
        return _portable_path(path)
    try:
        return _portable_path(path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return _portable_path(path)


class ProcessingStatus(StrEnum):
    """Machine-readable pipeline outcomes."""

    SUCCESS = "success"
    NO_PERSON = "no_person"
    INVALID_INPUT = "invalid_input"
    MODEL_ERROR = "model_error"
    PROCESSING_ERROR = "processing_error"
    PARTIAL_SUCCESS = "partial_success"


@dataclass(frozen=True, slots=True)
class CutoutOptions:
    """Tunable settings with conservative MVP defaults."""

    confidence_threshold: float = 0.70
    mask_threshold: float = 0.50
    min_area_ratio: float = 0.01
    max_people: int = 5
    max_input_side: int = 1920
    max_input_pixels: int = 40_000_000
    feather_radius: float = 1.5
    crop_padding_ratio: float = 0.04
    reject_severely_clipped: bool = True
    alpha_mode: Literal["hard", "soft"] = "soft"
    subject_mode: Literal["none", "people", "foreground"] = "people"

    def __post_init__(self) -> None:
        for name in ("confidence_threshold", "mask_threshold", "min_area_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_people < 1:
            raise ValueError("max_people must be at least 1")
        if self.max_input_side < 256:
            raise ValueError("max_input_side must be at least 256")
        if self.max_input_pixels < 1:
            raise ValueError("max_input_pixels must be positive")
        if self.feather_radius < 0:
            raise ValueError("feather_radius cannot be negative")
        if not 0.0 <= self.crop_padding_ratio <= 0.5:
            raise ValueError("crop_padding_ratio must be between 0 and 0.5")
        if self.alpha_mode not in {"hard", "soft"}:
            raise ValueError("alpha_mode must be 'hard' or 'soft'")
        if self.subject_mode not in {"none", "people", "foreground"}:
            raise ValueError("subject_mode must be 'none', 'people' or 'foreground'")


@dataclass(frozen=True, slots=True)
class PersonCutout:
    """Metadata for one exported person instance."""

    person_id: int
    confidence: float
    source_box: tuple[int, int, int, int]
    output_path: Path
    pixel_area: int

    def to_dict(self, output_dir: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        data["source_box"] = list(self.source_box)
        data["output_path"] = _artifact_path(self.output_path, output_dir)
        return data


@dataclass(frozen=True, slots=True)
class SubjectCutout:
    """Metadata for one template-ready person group."""

    subject_id: int
    member_person_ids: tuple[int, ...]
    mode: Literal["people", "foreground"]
    source_box: tuple[int, int, int, int]
    output_path: Path
    pixel_area: int

    def to_dict(self, output_dir: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        data["member_person_ids"] = list(self.member_person_ids)
        data["source_box"] = list(self.source_box)
        data["output_path"] = _artifact_path(self.output_path, output_dir)
        return data


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Complete result returned by both Python API and CLI."""

    status: ProcessingStatus
    input_path: Path
    output_dir: Path
    original_size: tuple[int, int] | None = None
    processed_size: tuple[int, int] | None = None
    people: tuple[PersonCutout, ...] = ()
    preview_path: Path | None = None
    manifest_path: Path | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    subjects: tuple[SubjectCutout, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": self.status.value,
            "input_path": _portable_path(self.input_path),
            "output_dir": _portable_path(self.output_dir),
            "original_size": list(self.original_size) if self.original_size else None,
            "processed_size": list(self.processed_size) if self.processed_size else None,
            "people": [person.to_dict(self.output_dir) for person in self.people],
            "subjects": [subject.to_dict(self.output_dir) for subject in self.subjects],
            "preview_path": (
                _artifact_path(self.preview_path, self.output_dir) if self.preview_path else None
            ),
            "manifest_path": (
                _artifact_path(self.manifest_path, self.output_dir) if self.manifest_path else None
            ),
            "warnings": list(self.warnings),
            "error": self.error,
            "timing_ms": self.timing_ms,
        }
