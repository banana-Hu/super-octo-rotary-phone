"""Stable input and output contracts for the cutout pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True, slots=True)
class PersonCutout:
    """Metadata for one exported person instance."""

    person_id: int
    confidence: float
    source_box: tuple[int, int, int, int]
    output_path: Path
    pixel_area: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_box"] = list(self.source_box)
        data["output_path"] = str(self.output_path)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "original_size": list(self.original_size) if self.original_size else None,
            "processed_size": list(self.processed_size) if self.processed_size else None,
            "people": [person.to_dict() for person in self.people],
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "warnings": list(self.warnings),
            "error": self.error,
            "timing_ms": self.timing_ms,
        }
