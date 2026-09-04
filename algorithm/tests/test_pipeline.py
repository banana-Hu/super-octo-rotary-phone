import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from momentmaker_cv.contracts import CutoutOptions, ProcessingStatus
from momentmaker_cv.pipeline import process_image
from momentmaker_cv.segmenter import MaskPrediction


class FakeSegmenter:
    def __init__(self, predictions: list[MaskPrediction]) -> None:
        self.predictions = predictions

    def predict(self, image: Image.Image) -> list[MaskPrediction]:
        return self.predictions


class BrokenSegmenter:
    def predict(self, image: Image.Image) -> list[MaskPrediction]:
        raise ValueError("invalid model output")


def test_pipeline_exports_transparent_people_preview_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "group.jpg"
    Image.new("RGB", (100, 80), "#336699").save(source)
    mask = np.zeros((80, 100), dtype=np.float32)
    mask[10:70, 20:60] = 1.0
    segmenter = FakeSegmenter([MaskPrediction(score=0.96, box=(20, 10, 60, 70), mask=mask)])

    result = process_image(
        source,
        tmp_path / "output",
        CutoutOptions(min_area_ratio=0, feather_radius=0),
        segmenter,
    )

    assert result.status is ProcessingStatus.SUCCESS
    assert len(result.people) == 1
    assert result.people[0].output_path.exists()
    assert result.preview_path and result.preview_path.exists()
    assert result.manifest_path and result.manifest_path.exists()
    with Image.open(result.people[0].output_path) as cutout:
        assert cutout.mode == "RGBA"
        assert cutout.getchannel("A").getextrema() == (0, 255)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert len(manifest["people"]) == 1


def test_pipeline_returns_no_person_and_still_exports_manifest(tmp_path: Path) -> None:
    source = tmp_path / "empty.png"
    Image.new("RGB", (50, 50), "white").save(source)

    result = process_image(source, tmp_path / "output", segmenter=FakeSegmenter([]))

    assert result.status is ProcessingStatus.NO_PERSON
    assert result.people == ()
    assert result.manifest_path and result.manifest_path.exists()


def test_pipeline_reports_invalid_input_without_running_model(tmp_path: Path) -> None:
    result = process_image(
        tmp_path / "missing.jpg",
        tmp_path / "output",
        segmenter=FakeSegmenter([]),
    )

    assert result.status is ProcessingStatus.INVALID_INPUT
    assert result.error and "does not exist" in result.error


def test_pipeline_converts_unexpected_model_exception_to_result(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (50, 50), "white").save(source)

    result = process_image(source, tmp_path / "output", segmenter=BrokenSegmenter())

    assert result.status is ProcessingStatus.MODEL_ERROR
    assert result.error == "invalid model output"


def test_pipeline_converts_export_exception_to_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (50, 50), "white").save(source)

    def fail_export(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk is read-only")

    monkeypatch.setattr("momentmaker_cv.pipeline.export_people", fail_export)
    result = process_image(source, tmp_path / "output", segmenter=FakeSegmenter([]))

    assert result.status is ProcessingStatus.PROCESSING_ERROR
    assert result.error == "artifact export failed: disk is read-only"


def test_pipeline_returns_partial_success_when_manifest_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (50, 50), "white").save(source)

    def fail_manifest(*_args: object, **_kwargs: object) -> None:
        raise OSError("manifest is locked")

    monkeypatch.setattr("momentmaker_cv.pipeline.export_manifest", fail_manifest)
    result = process_image(source, tmp_path / "output", segmenter=FakeSegmenter([]))

    assert result.status is ProcessingStatus.PARTIAL_SUCCESS
    assert result.preview_path and result.preview_path.exists()
    assert result.manifest_path is None
    assert result.error == "manifest export failed: manifest is locked"
