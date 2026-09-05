"""Optional salient-foreground adapter used to retain person-connected objects."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any, Protocol

import numpy as np
from PIL import Image


class ForegroundModelUnavailableError(RuntimeError):
    """Raised when the optional foreground runtime cannot be initialized."""


class ForegroundSegmenter(Protocol):
    def predict(self, image: Image.Image) -> np.ndarray: ...


class InSPyReNetForegroundSegmenter:
    """Lazy adapter for transparent-background's InSPyReNet map output."""

    def __init__(
        self,
        device: str = "cpu",
        remover_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.device = device
        self._remover_factory = remover_factory
        self._remover: Any = None
        self._lock = RLock()

    def _load(self) -> Any:
        with self._lock:
            if self._remover is not None:
                return self._remover
            try:
                if self._remover_factory is None:
                    from transparent_background import Remover

                    remover = Remover(device=self.device)
                else:
                    remover = self._remover_factory(self.device)
            except Exception as exc:
                raise ForegroundModelUnavailableError(
                    "InSPyReNet is unavailable. Install the foreground extra with "
                    "`pip install -e .[foreground]`."
                ) from exc
            self._remover = remover
            return remover

    def predict(self, image: Image.Image) -> np.ndarray:
        with self._lock:
            remover = self._load()
            try:
                output = remover.process(image, type="map")
                values = np.asarray(
                    output.convert("L") if isinstance(output, Image.Image) else output
                )
            except Exception as exc:
                raise RuntimeError(f"InSPyReNet inference failed: {exc}") from exc

        if values.ndim != 2:
            raise RuntimeError("foreground model output must be a two-dimensional mask")
        if values.shape != (image.height, image.width):
            raise RuntimeError(
                "foreground model output shape does not match the processed image: "
                f"{values.shape} != {(image.height, image.width)}"
            )
        probability = values.astype(np.float32)
        if probability.size and float(probability.max()) > 1.0:
            probability /= 255.0
        return np.clip(probability, 0.0, 1.0)
