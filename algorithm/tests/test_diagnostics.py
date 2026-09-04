import json
from pathlib import Path

import numpy as np
from PIL import Image

from momentmaker_cv.contracts import PersonCutout, ProcessingResult, ProcessingStatus
from momentmaker_cv.diagnostics import check_model, validate_result_artifacts
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


def test_validate_result_artifacts_accepts_complete_result(tmp_path: Path) -> None:
    output = tmp_path / "output"
    person_path = output / "people" / "person_01.png"
    person_path.parent.mkdir(parents=True)
    rgba = Image.new("RGBA", (20, 30), (20, 40, 60, 255))
    alpha = np.zeros((30, 20), dtype=np.uint8)
    alpha[4:26, 5:15] = 255
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    rgba.save(person_path)
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
