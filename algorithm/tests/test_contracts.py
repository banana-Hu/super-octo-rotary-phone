from pathlib import Path

import pytest

from momentmaker_cv.contracts import (
    CutoutOptions,
    PersonCutout,
    ProcessingResult,
    ProcessingStatus,
)


def test_options_reject_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        CutoutOptions(confidence_threshold=1.1)


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
    )

    payload = result.to_dict()

    assert payload["status"] == "success"
    assert payload["people"][0]["output_path"] == "people\\person_01.png"
    assert payload["people"][0]["source_box"] == [10, 20, 110, 220]
