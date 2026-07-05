from bardecode.decoder import decode_region
from bardecode.image_io import read_image


def test_decode_synthetic_ean13(synthetic_ean13_image):
    img = read_image(synthetic_ean13_image)
    results = decode_region(img)
    assert len(results) >= 1
    assert any(r.format.upper().startswith("EAN") for r in results)
    assert any("4006381333917" in r.text for r in results)


def test_decode_blank_returns_empty(blank_image):
    img = read_image(blank_image)
    assert decode_region(img) == []
