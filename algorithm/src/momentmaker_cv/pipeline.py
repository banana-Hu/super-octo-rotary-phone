"""Public orchestration API for person cutout processing."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .contracts import CutoutOptions, ProcessingResult, ProcessingStatus
from .exporter import (
    cleanup_stale_people,
    cleanup_stale_subjects,
    export_manifest,
    export_people,
    export_subjects,
)
from .foreground import ForegroundSegmenter, InSPyReNetForegroundSegmenter
from .image_io import InvalidImageError, load_image
from .mask_processing import process_predictions
from .preview import create_preview
from .segmenter import PersonSegmenter, TorchvisionMaskRCNNSegmenter
from .subject_processing import build_subject_masks

_DEFAULT_SEGMENTER = TorchvisionMaskRCNNSegmenter()
_DEFAULT_FOREGROUND_SEGMENTER = InSPyReNetForegroundSegmenter()


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def process_image(
    input_path: str | Path,
    output_dir: str | Path,
    options: CutoutOptions | None = None,
    segmenter: PersonSegmenter | None = None,
    foreground_segmenter: ForegroundSegmenter | None = None,
) -> ProcessingResult:
    """Extract individual people and optional template-ready subject groups."""

    source = Path(input_path)
    destination = Path(output_dir)
    settings = options or CutoutOptions()
    timing: dict[str, float] = {}

    started = perf_counter()
    try:
        loaded = load_image(
            source,
            max_side=settings.max_input_side,
            max_pixels=settings.max_input_pixels,
        )
    except InvalidImageError as exc:
        return ProcessingResult(
            status=ProcessingStatus.INVALID_INPUT,
            input_path=source,
            output_dir=destination,
            error=str(exc),
            timing_ms={"total": _elapsed_ms(started)},
        )
    timing["load"] = _elapsed_ms(started)

    inference_started = perf_counter()
    detector = segmenter if segmenter is not None else _DEFAULT_SEGMENTER
    try:
        predictions = detector.predict(loaded.image)
    except Exception as exc:
        timing["inference"] = _elapsed_ms(inference_started)
        timing["total"] = _elapsed_ms(started)
        return ProcessingResult(
            status=ProcessingStatus.MODEL_ERROR,
            input_path=source,
            output_dir=destination,
            original_size=loaded.original_size,
            processed_size=loaded.processed_size,
            error=str(exc),
            timing_ms=timing,
        )
    timing["inference"] = _elapsed_ms(inference_started)

    post_started = perf_counter()
    try:
        masks = process_predictions(loaded.processed_size, predictions, settings)
    except Exception as exc:
        timing["postprocess"] = _elapsed_ms(post_started)
        timing["total"] = _elapsed_ms(started)
        return ProcessingResult(
            status=ProcessingStatus.PROCESSING_ERROR,
            input_path=source,
            output_dir=destination,
            original_size=loaded.original_size,
            processed_size=loaded.processed_size,
            error=f"post-processing failed: {exc}",
            timing_ms=timing,
        )
    timing["postprocess"] = _elapsed_ms(post_started)

    foreground_warning: str | None = None
    foreground = None
    if settings.subject_mode == "foreground" and masks:
        foreground_started = perf_counter()
        enhancer = (
            foreground_segmenter
            if foreground_segmenter is not None
            else _DEFAULT_FOREGROUND_SEGMENTER
        )
        try:
            foreground = enhancer.predict(loaded.image)
        except Exception as exc:
            foreground_warning = f"Foreground enhancement failed; used people-only subjects: {exc}"
        timing["foreground"] = _elapsed_ms(foreground_started)

    subject_started = perf_counter()
    try:
        subject_masks = (
            build_subject_masks(loaded.processed_size, masks, foreground)
            if settings.subject_mode != "none"
            else []
        )
    except Exception as exc:
        timing["subject_postprocess"] = _elapsed_ms(subject_started)
        timing["total"] = _elapsed_ms(started)
        return ProcessingResult(
            status=ProcessingStatus.PROCESSING_ERROR,
            input_path=source,
            output_dir=destination,
            original_size=loaded.original_size,
            processed_size=loaded.processed_size,
            error=f"subject post-processing failed: {exc}",
            timing_ms=timing,
        )
    timing["subject_postprocess"] = _elapsed_ms(subject_started)

    export_started = perf_counter()
    try:
        destination.mkdir(parents=True, exist_ok=True)
        people, cutout_images = export_people(loaded.image, masks, destination)
        subjects = export_subjects(loaded.image, subject_masks, destination)
        preview_path = destination / "preview.png"
        create_preview(cutout_images, preview_path)
        cleanup_warnings = (
            *cleanup_stale_people(destination, people),
            *cleanup_stale_subjects(destination, subjects),
        )
    except Exception as exc:
        timing["export"] = _elapsed_ms(export_started)
        timing["total"] = _elapsed_ms(started)
        return ProcessingResult(
            status=ProcessingStatus.PROCESSING_ERROR,
            input_path=source,
            output_dir=destination,
            original_size=loaded.original_size,
            processed_size=loaded.processed_size,
            error=f"artifact export failed: {exc}",
            timing_ms=timing,
        )
    timing["export"] = _elapsed_ms(export_started)
    timing["total"] = _elapsed_ms(started)

    manifest_path = destination / "result.json"
    result_warnings = list(cleanup_warnings)
    if foreground_warning:
        result_warnings.insert(0, foreground_warning)
    if not people:
        result_warnings.insert(0, "No person passed the quality filters.")
    result = ProcessingResult(
        status=(ProcessingStatus.SUCCESS if people else ProcessingStatus.NO_PERSON),
        input_path=source,
        output_dir=destination,
        original_size=loaded.original_size,
        processed_size=loaded.processed_size,
        people=people,
        subjects=subjects,
        preview_path=preview_path,
        manifest_path=manifest_path,
        warnings=tuple(result_warnings),
        timing_ms=timing,
    )
    try:
        export_manifest(result, manifest_path)
        return result
    except Exception as exc:
        return ProcessingResult(
            status=ProcessingStatus.PARTIAL_SUCCESS,
            input_path=source,
            output_dir=destination,
            original_size=loaded.original_size,
            processed_size=loaded.processed_size,
            people=people,
            subjects=subjects,
            preview_path=preview_path,
            warnings=(
                *result_warnings,
                "People, subjects and preview were exported, but result.json was not written.",
            ),
            error=f"manifest export failed: {exc}",
            timing_ms=timing,
        )
