import numpy as np
import pytest
from PIL import Image

from momentmaker_cv.foreground import (
    ForegroundModelUnavailableError,
    InSPyReNetForegroundSegmenter,
)


class FakeRemover:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, image: Image.Image, *, type: str) -> Image.Image:
        self.calls += 1
        assert type == "map"
        return Image.new("RGB", image.size, (128, 128, 128))


def test_adapter_loads_once_and_returns_normalized_probability() -> None:
    remover = FakeRemover()
    factory_calls: list[str] = []

    def factory(device: str) -> FakeRemover:
        factory_calls.append(device)
        return remover

    adapter = InSPyReNetForegroundSegmenter(device="cpu", remover_factory=factory)
    image = Image.new("RGB", (8, 6), "white")

    first = adapter.predict(image)
    second = adapter.predict(image)

    assert factory_calls == ["cpu"]
    assert remover.calls == 2
    assert first.shape == (6, 8)
    assert first.dtype == np.float32
    assert np.allclose(first, 128 / 255)
    assert np.array_equal(first, second)


def test_adapter_rejects_invalid_map_shape() -> None:
    class BadRemover:
        def process(self, image: Image.Image, *, type: str) -> np.ndarray:
            return np.zeros((2, 2, 2), dtype=np.uint8)

    adapter = InSPyReNetForegroundSegmenter(remover_factory=lambda _device: BadRemover())

    with pytest.raises(RuntimeError, match="two-dimensional"):
        adapter.predict(Image.new("RGB", (8, 6)))


def test_adapter_wraps_loader_failure() -> None:
    def broken_factory(_device: str) -> object:
        raise ImportError("missing optional dependency")

    adapter = InSPyReNetForegroundSegmenter(remover_factory=broken_factory)

    with pytest.raises(ForegroundModelUnavailableError, match="foreground extra"):
        adapter.predict(Image.new("RGB", (8, 6)))
