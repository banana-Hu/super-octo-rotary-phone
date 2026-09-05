import json
from pathlib import Path

import numpy as np
from PIL import Image

from momentmaker_cv.contracts import (
    PersonCutout,
    ProcessingResult,
    ProcessingStatus,
    SubjectCutout,
)
from momentmaker_cv.diagnostics import build_parser, check_model, validate_result_artifacts
from momentmaker_cv.segmenter import MaskPrediction


class RecordingSegmenter:
    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    @property
    def active_device(self) -> str:
        return "cpu"

    def predict(self, image: Image.Image) -> list[MaskPrediction]:
        self.sizes.append(image.size)
        return []


class BrokenSegmenter:
    @property
    def active_device(self) -> None:
        return None

    def predict(self, image: Image.Image) -> list[MaskPrediction]:
        raise RuntimeError("weights are unavailable")


class RecordingForegroundSegmenter:
    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def predict(self, image: Image.Image) -> np.ndarray:
        self.sizes.append(image.size)
        return np.zeros((image.height, image.width), dtype=np.float32)


def test_check_model_runs_a_minimal_inference() -> None:
    segmenter = RecordingSegmenter()

    report = check_model(segmenter=segmenter)

    assert report["status"] == "ready"
    assert report["device"] == "cpu"
    assert report["smoke_image_size"] == [64, 64]
    assert report["timing_ms"] >= 0
    assert segmenter.sizes == [(64, 64)]


def test_check_model_returns_actionable_failure() -> None:
    report = check_model(segmenter=BrokenSegmenter())

    assert report["status"] == "error"
    assert report["error"] == "weights are unavailable"


def test_check_model_can_verify_foreground_runtime() -> None:
    foreground = RecordingForegroundSegmenter()

    report = check_model(
        segmenter=RecordingSegmenter(),
        foreground_segmenter=foreground,
    )

    assert report["status"] == "ready"
    assert report["foreground"] == "ready"
    assert foreground.sizes == [(64, 64)]


def test_diagnostics_parser_accepts_foreground_subject_mode() -> None:
    args = build_parser().parse_args(["--subject-mode", "foreground"])

    assert args.subject_mode == "foreground"


def test_validate_result_artifacts_accepts_complete_result(tmp_path: Path) -> None:
    output = tmp_path / "output"
    person_path = output / "people" / "person_01.png"
    person_path.parent.mkdir(parents=True)
    rgba = Image.new("RGBA", (20, 30), (20, 40, 60, 255))
    alpha = np.zeros((30, 20), dtype=np.uint8)
    alpha[4:26, 5:15] = 255
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    rgba.save(person_path)
    subject_path = output / "subjects" / "subject_01.png"
    subject_path.parent.mkdir(parents=True)
    rgba.save(subject_path)
    preview_path = output / "preview.png"
    Image.new("RGB", (100, 50), "white").save(preview_path)
    manifest_path = output / "result.json"
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        input_path=tmp_path / "source.jpg",
        output_dir=output,
        original_size=(100, 80),
        processed_size=(100, 80),
        people=(
            PersonCutout(
                person_id=1,
                confidence=0.95,
                source_box=(10, 10, 30, 60),
                output_path=person_path,
                pixel_area=220,
            ),
        ),
        subjects=(
            SubjectCutout(
                subject_id=1,
                member_person_ids=(1,),
                mode="people",
                source_box=(10, 10, 30, 60),
                output_path=subject_path,
                pixel_area=220,
            ),
        ),
        primary_subject_id=1,
        preview_path=preview_path,
        manifest_path=manifest_path,
    )
    manifest_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    assert validate_result_artifacts(result) == []


def test_validate_result_artifacts_reports_missing_and_opaque_outputs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    person_path = output / "people" / "person_01.png"
    person_path.parent.mkdir(parents=True)
    Image.new("RGBA", (20, 30), (20, 40, 60, 255)).save(person_path)
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        input_path=tmp_path / "source.jpg",
        output_dir=output,
        people=(
            PersonCutout(
                person_id=1,
                confidence=0.95,
                source_box=(10, 10, 30, 60),
                output_path=person_path,
                pixel_area=220,
            ),
        ),
        preview_path=output / "preview.png",
        manifest_path=output / "result.json",
    )

    errors = validate_result_artifacts(result)

    assert "preview.png is missing" in errors
    assert "result.json is missing" in errors
    assert "people/person_01.png has no transparent pixels" in errors
    assert "no subject cutouts were produced" in errors


def test_validate_result_artifacts_rejects_invalid_primary_subject(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    manifest_path = output / "result.json"
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        input_path=tmp_path / "source.jpg",
        output_dir=output,
        primary_subject_id=9,
        manifest_path=manifest_path,
    )
    manifest_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    errors = validate_result_artifacts(result)

    assert "primary_subject_id does not reference an exported subject" in errors


def test_validate_result_artifacts_allows_no_subject_mode(tmp_path: Path) -> None:
    result = ProcessingResult(
        status=ProcessingStatus.NO_PERSON,
        input_path=tmp_path / "source.jpg",
        output_dir=tmp_path / "output",
    )

    errors = validate_result_artifacts(result, require_subjects=False)

    assert "no subject cutouts were produced" not in errors


def test_validate_result_artifacts_detects_foreground_fallback(tmp_path: Path) -> None:
    subject = SubjectCutout(
        subject_id=1,
        member_person_ids=(1,),
        mode="people",
        source_box=(0, 0, 10, 10),
        output_path=tmp_path / "missing.png",
        pixel_area=100,
    )
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        input_path=tmp_path / "source.jpg",
        output_dir=tmp_path,
        subjects=(subject,),
        primary_subject_id=1,
    )

    errors = validate_result_artifacts(result, require_foreground=True)

    assert "foreground enhancement was requested but not applied" in errors


def test_validate_result_artifacts_handles_malformed_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    manifest_path = output / "result.json"
    manifest_path.write_text("[]", encoding="utf-8")
    result = ProcessingResult(
        status=ProcessingStatus.NO_PERSON,
        input_path=tmp_path / "source.jpg",
        output_dir=output,
        manifest_path=manifest_path,
    )

    errors = validate_result_artifacts(result)

    assert "result.json must contain a JSON object" in errors
