import numpy as np

from momentmaker_cv.contracts import CutoutOptions
from momentmaker_cv.mask_processing import process_predictions
from momentmaker_cv.segmenter import MaskPrediction


def _prediction(
    score: float,
    mask: np.ndarray,
    box: tuple[float, float, float, float] = (1.2, 2.1, 8.8, 9.9),
) -> MaskPrediction:
    return MaskPrediction(score=score, box=box, mask=mask)


def test_process_predictions_filters_score_and_small_area() -> None:
    large = np.zeros((10, 10), dtype=np.float32)
    large[2:9, 2:8] = 1.0
    tiny = np.zeros((10, 10), dtype=np.float32)
    tiny[4, 4] = 1.0
    options = CutoutOptions(
        confidence_threshold=0.7,
        min_area_ratio=0.1,
        feather_radius=0,
        crop_padding_ratio=0,
    )

    results = process_predictions(
        (10, 10),
        [_prediction(0.5, large), _prediction(0.9, tiny), _prediction(0.8, large)],
        options,
    )

    assert len(results) == 1
    assert results[0].score == 0.8
    # Morphological closing may extend an edge by one pixel; the crop must still
    # tightly contain the cleaned mask.
    assert results[0].crop_box == (2, 2, 8, 10)
    assert results[0].source_box == (1, 2, 9, 10)


def test_process_predictions_resizes_masks_and_limits_count() -> None:
    mask = np.ones((4, 4), dtype=np.float32)
    options = CutoutOptions(max_people=2, min_area_ratio=0, feather_radius=0)

    results = process_predictions(
        (8, 12),
        [_prediction(0.8, mask), _prediction(0.9, mask), _prediction(0.85, mask)],
        options,
    )

    assert [item.score for item in results] == [0.9, 0.85]
    assert results[0].alpha.shape == (12, 8)


def test_feathering_produces_soft_alpha_edges() -> None:
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:15, 5:15] = 1.0

    result = process_predictions(
        (20, 20),
        [_prediction(0.9, mask)],
        CutoutOptions(min_area_ratio=0, feather_radius=1.5, alpha_mode="hard"),
    )[0]

    assert np.any((result.alpha > 0) & (result.alpha < 255))


def test_rejects_narrow_person_fragment_clipped_by_side_edge() -> None:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:90, 0:8] = 1.0
    prediction = _prediction(0.95, mask, box=(0, 20, 8, 90))

    results = process_predictions(
        (100, 100),
        [prediction],
        CutoutOptions(min_area_ratio=0, feather_radius=0),
    )

    assert results == []


def test_keeps_partial_person_when_filter_is_disabled() -> None:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:90, 0:8] = 1.0
    prediction = _prediction(0.95, mask, box=(0, 20, 8, 90))

    results = process_predictions(
        (100, 100),
        [prediction],
        CutoutOptions(
            min_area_ratio=0,
            feather_radius=0,
            reject_severely_clipped=False,
        ),
    )

    assert len(results) == 1


def test_keeps_person_touching_only_bottom_edge() -> None:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:100, 40:50] = 1.0
    prediction = _prediction(0.95, mask, box=(40, 20, 50, 100))

    results = process_predictions(
        (100, 100),
        [prediction],
        CutoutOptions(min_area_ratio=0, feather_radius=0),
    )

    assert len(results) == 1


def test_keeps_wide_person_even_when_touching_side_edge() -> None:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:90, 0:30] = 1.0
    prediction = _prediction(0.95, mask, box=(0, 10, 30, 90))

    results = process_predictions(
        (100, 100),
        [prediction],
        CutoutOptions(min_area_ratio=0, feather_radius=0),
    )

    assert len(results) == 1


def test_selected_people_are_ordered_from_left_to_right() -> None:
    mask = np.ones((100, 100), dtype=np.float32)
    predictions = [
        _prediction(0.99, mask, box=(70, 10, 90, 90)),
        _prediction(0.90, mask, box=(10, 10, 30, 90)),
        _prediction(0.95, mask, box=(40, 10, 60, 90)),
    ]

    results = process_predictions(
        (100, 100),
        predictions,
        CutoutOptions(min_area_ratio=0, feather_radius=0),
    )

    assert [item.score for item in results] == [0.90, 0.95, 0.99]


def test_people_limit_is_applied_before_spatial_ordering() -> None:
    mask = np.ones((100, 100), dtype=np.float32)
    predictions = [
        _prediction(0.60, mask, box=(5, 10, 25, 90)),
        _prediction(0.95, mask, box=(35, 10, 55, 90)),
        _prediction(0.90, mask, box=(65, 10, 85, 90)),
    ]

    results = process_predictions(
        (100, 100),
        predictions,
        CutoutOptions(
            confidence_threshold=0.5,
            max_people=2,
            min_area_ratio=0,
            feather_radius=0,
        ),
    )

    assert [item.score for item in results] == [0.95, 0.90]


def test_soft_alpha_keeps_smooth_transition_near_clean_mask() -> None:
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[4:16, 4:16] = 0.35
    mask[5:15, 5:15] = 0.50
    mask[6:14, 6:14] = 0.80

    result = process_predictions(
        (20, 20),
        [_prediction(0.9, mask, box=(4, 4, 16, 16))],
        CutoutOptions(min_area_ratio=0, alpha_mode="soft"),
    )[0]

    assert result.alpha[0, 0] == 0
    assert 0 < result.alpha[5, 5] < 255
    assert result.alpha[10, 10] == 255


def test_hard_alpha_remains_binary_when_feathering_is_disabled() -> None:
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:15, 5:15] = 0.8

    result = process_predictions(
        (20, 20),
        [_prediction(0.9, mask, box=(5, 5, 15, 15))],
        CutoutOptions(min_area_ratio=0, feather_radius=0, alpha_mode="hard"),
    )[0]

    assert set(np.unique(result.alpha)) == {0, 255}
