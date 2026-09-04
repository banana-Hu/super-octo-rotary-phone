"""Person instance-segmentation adapters.

Torch imports and model construction are deliberately lazy so tests and non-model
code remain fast. The default adapter uses torchvision's BSD-licensed Mask R-CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

import numpy as np
from PIL import Image


class ModelUnavailableError(RuntimeError):
    """Raised when the optional model runtime cannot be loaded."""


@dataclass(frozen=True, slots=True)
class MaskPrediction:
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray


class PersonSegmenter(Protocol):
    def predict(self, image: Image.Image) -> list[MaskPrediction]: ...


def _to_numpy(value: Any) -> np.ndarray:
    for method in ("detach", "cpu"):
        operation = getattr(value, method, None)
        if operation is not None:
            value = operation()
    to_array = getattr(value, "numpy", None)
    return np.asarray(to_array() if to_array is not None else value)


def predictions_from_tensors(
    labels: Any,
    scores: Any,
    boxes: Any,
    masks: Any,
    *,
    person_label: int = 1,
) -> list[MaskPrediction]:
    """Convert a torchvision-style output to person-only NumPy predictions."""

    label_values = _to_numpy(labels).reshape(-1)
    score_values = _to_numpy(scores).reshape(-1)
    box_values = _to_numpy(boxes).reshape(-1, 4)
    mask_values = _to_numpy(masks)
    if mask_values.ndim == 4 and mask_values.shape[1] == 1:
        mask_values = mask_values[:, 0]
    if mask_values.ndim != 3:
        raise ValueError("masks must have shape [N, H, W] or [N, 1, H, W]")

    lengths = {len(label_values), len(score_values), len(box_values), len(mask_values)}
    if len(lengths) != 1:
        raise ValueError("model output arrays must contain the same number of items")

    predictions = [
        MaskPrediction(
            score=float(score_values[index]),
            box=tuple(float(value) for value in box_values[index]),
            mask=np.asarray(mask_values[index], dtype=np.float32),
        )
        for index, label in enumerate(label_values)
        if int(label) == person_label
    ]
    return sorted(predictions, key=lambda item: item.score, reverse=True)


class TorchvisionMaskRCNNSegmenter:
    """Lazy Mask R-CNN adapter that supports CPU and CUDA when available."""

    def __init__(self, device: str | None = None) -> None:
        self.requested_device = device
        self._device: str | None = None
        self._model: Any = None
        self._transform: Any = None
        self._lock = RLock()

    @property
    def active_device(self) -> str | None:
        """Device selected after model initialization, if available."""

        return self._device

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from torchvision.models.detection import (
                    MaskRCNN_ResNet50_FPN_V2_Weights,
                    maskrcnn_resnet50_fpn_v2,
                )
            except (ImportError, OSError) as exc:
                raise ModelUnavailableError(
                    "Mask R-CNN is unavailable. Install the model extra with "
                    "`pip install -e .[model]`."
                ) from exc

            if self.requested_device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.requested_device
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise ModelUnavailableError("CUDA was requested but is not available")

            weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            try:
                model = maskrcnn_resnet50_fpn_v2(weights=weights)
                model.eval().to(device)
            except Exception as exc:
                raise ModelUnavailableError(f"could not initialize Mask R-CNN: {exc}") from exc

            self._device = device
            self._model = model
            self._transform = weights.transforms()

    def predict(self, image: Image.Image) -> list[MaskPrediction]:
        with self._lock:
            self._load()
            try:
                import torch

                tensor = self._transform(image).to(self._device)
                with torch.inference_mode():
                    output = self._model([tensor])[0]
                return predictions_from_tensors(
                    output["labels"], output["scores"], output["boxes"], output["masks"]
                )
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Mask R-CNN inference failed: {exc}") from exc
