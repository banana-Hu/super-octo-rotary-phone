"""Public orchestration API for person cutout processing."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .contracts import CutoutOptions, ProcessingResult, ProcessingStatus
from .exporter import export_manifest, export_people
from .image_io import InvalidImageError, load_image
from .mask_processing import process_predictions
from .preview import create_preview
from .segmenter import (
    ModelUnavailableError,
    PersonSegmenter,
    TorchvisionMaskRCNNSegmenter,
)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def process_image(
    input_path: str | Path,
    output_dir: str | Path,
    options: CutoutOptions | None = None,
    segmenter: PersonSegmenter | None = None,
) -> ProcessingResult:
    """Extract individual people and write transparent PNG artifacts."""

    source = Path(input_path)
    destination = Path(output_dir)
    settings = options or CutoutOptions()
    timing: dict[str, float] = {}

    started = perf_counter()
    try:
        loaded = load_image(source, max_side=settings.max_input_side)
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
    detector = segmenter or TorchvisionMaskRCNNSegmenter()
    try:
        predictions = detector.predict(loaded.image)
    except (ModelUnavailableError, RuntimeError) as exc:
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
    masks = process_predictions(loaded.processed_size, predictions, settings)
    timing["postprocess"] = _elapsed_ms(post_started)

    export_started = perf_counter()
    destination.mkdir(parents=True, exist_ok=True)
    people, cutout_images = export_people(loaded.image, masks, destination)
    preview_path = destination / "preview.png"
    create_preview(cutout_images, preview_path)
    timing["export"] = _elapsed_ms(export_started)
    timing["total"] = _elapsed_ms(started)

    manifest_path = destination / "result.json"
    result = ProcessingResult(
        status=(ProcessingStatus.SUCCESS if people else ProcessingStatus.NO_PERSON),
        input_path=source,
        output_dir=destination,
        original_size=loaded.original_size,
        processed_size=loaded.processed_size,
        people=people,
        preview_path=preview_path,
        manifest_path=manifest_path,
        warnings=("No person passed the quality filters.",) if not people else (),
        timing_ms=timing,
    )
    export_manifest(result, manifest_path)
    return result
