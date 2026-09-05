from pathlib import Path

import pytest

from momentmaker_cv.contracts import (
    CutoutOptions,
    PersonCutout,
    ProcessingResult,
    ProcessingStatus,
    SubjectCutout,
)


def test_options_reject_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        CutoutOptions(confidence_threshold=1.1)


def test_options_reject_invalid_alpha_mode() -> None:
    with pytest.raises(ValueError, match="alpha_mode"):
        CutoutOptions(alpha_mode="unknown")  # type: ignore[arg-type]


def test_options_use_soft_alpha_by_default() -> None:
    assert CutoutOptions().alpha_mode == "soft"


def test_options_reject_invalid_subject_mode() -> None:
    with pytest.raises(ValueError, match="subject_mode"):
        CutoutOptions(subject_mode="unknown")  # type: ignore[arg-type]


def test_result_serializes_paths_and_enums() -> None:
    person = PersonCutout(
        person_id=1,
        confidence=0.95,
        source_box=(10, 20, 110, 220),
        output_path=Path("people/person_01.png"),
        pixel_area=12_000,
    )
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        input_path=Path("input.jpg"),
        output_dir=Path("output"),
        original_size=(640, 480),
        processed_size=(640, 480),
        people=(person,),
        subjects=(
            SubjectCutout(
                subject_id=1,
                member_person_ids=(1,),
                mode="people",
                source_box=(10, 20, 110, 220),
                output_path=Path("subjects/subject_01.png"),
                pixel_area=12_000,
            ),
        ),
        primary_subject_id=1,
    )

    payload = result.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "success"
    assert payload["people"][0]["output_path"] == "people/person_01.png"
    assert payload["people"][0]["source_box"] == [10, 20, 110, 220]
    assert payload["subjects"][0]["member_person_ids"] == [1]
    assert payload["subjects"][0]["output_path"] == "subjects/subject_01.png"
    assert payload["primary_subject_id"] == 1


def test_result_serializes_artifacts_relative_to_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "job-123"
    person = PersonCutout(
        person_id=1,
        confidence=0.9,
        source_box=(0, 0, 10, 20),
        output_path=output_dir / "people" / "person_01.png",
        pixel_area=100,
    )
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        input_path=tmp_path / "input.jpg",
        output_dir=output_dir,
        people=(person,),
        preview_path=output_dir / "preview.png",
        manifest_path=output_dir / "result.json",
    )

    payload = result.to_dict()

    assert payload["people"][0]["output_path"] == "people/person_01.png"
    assert payload["preview_path"] == "preview.png"
    assert payload["manifest_path"] == "result.json"
