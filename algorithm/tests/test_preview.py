from pathlib import Path

from PIL import Image, ImageColor

from momentmaker_cv.preview import PREVIEW_DARK_BG, PREVIEW_LIGHT_BG, create_preview


def test_preview_cards_use_light_and_dark_backgrounds(tmp_path: Path) -> None:
    target = tmp_path / "preview.png"

    create_preview([], target)

    with Image.open(target) as preview:
        assert preview.getpixel((100, 300)) == ImageColor.getrgb(PREVIEW_LIGHT_BG)
        assert preview.getpixel((220, 300)) == ImageColor.getrgb(PREVIEW_DARK_BG)


def test_transparent_cutout_reveals_both_card_backgrounds(tmp_path: Path) -> None:
    target = tmp_path / "preview.png"
    cutout = Image.new("RGBA", (200, 200), (255, 0, 0, 0))
    for x in range(80, 120):
        for y in range(20, 180):
            cutout.putpixel((x, y), (255, 0, 0, 255))

    create_preview([cutout], target)

    with Image.open(target) as preview:
        assert preview.getpixel((100, 500)) == ImageColor.getrgb(PREVIEW_LIGHT_BG)
        assert preview.getpixel((220, 500)) == ImageColor.getrgb(PREVIEW_DARK_BG)
        assert preview.getpixel((165, 500)) == (255, 0, 0)
