from pathlib import Path

import pytest
from PIL import Image

from momentmaker_cv.image_io import InvalidImageError, load_image


def test_load_image_converts_to_rgb_and_downscales(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    Image.new("RGBA", (800, 400), (20, 40, 60, 128)).save(source)

    loaded = load_image(source, max_side=400)

    assert loaded.original_size == (800, 400)
    assert loaded.processed_size == (400, 200)
    assert loaded.image.mode == "RGB"


def test_load_image_does_not_upscale(tmp_path: Path) -> None:
    source = tmp_path / "small.jpg"
    Image.new("RGB", (320, 200), "white").save(source)

    loaded = load_image(source, max_side=640)

    assert loaded.processed_size == (320, 200)


def test_load_image_rejects_non_image(tmp_path: Path) -> None:
    source = tmp_path / "fake.png"
    source.write_text("not an image", encoding="utf-8")

    with pytest.raises(InvalidImageError, match="cannot read image"):
        load_image(source)


def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError, match="does not exist"):
        load_image(tmp_path / "missing.jpg")


def test_load_image_rejects_excessive_pixel_count_before_decode(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    Image.new("RGB", (20, 20), "white").save(source)

    with pytest.raises(InvalidImageError, match="too many pixels"):
        load_image(source, max_pixels=399)
