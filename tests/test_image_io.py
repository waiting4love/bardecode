import pytest
from bardecode.image_io import read_image


def test_read_png(png_image):
    img = read_image(png_image)
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_read_webp(webp_image):
    img = read_image(webp_image)
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_read_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(str(tmp_path / "nope.png"))
