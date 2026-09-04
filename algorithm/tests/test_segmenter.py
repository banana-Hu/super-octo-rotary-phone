import numpy as np
import pytest

from momentmaker_cv.segmenter import predictions_from_tensors


def test_predictions_keep_only_people_and_sort_by_score() -> None:
    predictions = predictions_from_tensors(
        labels=np.array([1, 18, 1]),
        scores=np.array([0.75, 0.99, 0.92]),
        boxes=np.array([[0, 0, 10, 20], [1, 1, 5, 5], [4, 2, 18, 25]]),
        masks=np.ones((3, 1, 8, 6), dtype=np.float32),
    )

    assert [prediction.score for prediction in predictions] == pytest.approx([0.92, 0.75])
    assert predictions[0].box == (4.0, 2.0, 18.0, 25.0)
    assert predictions[0].mask.shape == (8, 6)


def test_predictions_reject_mismatched_model_outputs() -> None:
    with pytest.raises(ValueError, match="same number"):
        predictions_from_tensors(
            labels=np.array([1, 1]),
            scores=np.array([0.8]),
            boxes=np.ones((2, 4)),
            masks=np.ones((2, 4, 4)),
        )


def test_predictions_reject_invalid_mask_shape() -> None:
    with pytest.raises(ValueError, match="masks must have shape"):
        predictions_from_tensors(
            labels=np.array([1]),
            scores=np.array([0.8]),
            boxes=np.ones((1, 4)),
            masks=np.ones((4, 4)),
        )
