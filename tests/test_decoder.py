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


def test_decode_with_formats_whitelist_match(synthetic_ean13_image):
    """白名单包含目标格式时仍能解码。"""
    img = read_image(synthetic_ean13_image)
    results = decode_region(img, formats=["EAN13"])
    assert len(results) >= 1
    assert any("4006381333917" in r.text for r in results)


def test_decode_with_formats_whitelist_excludes(synthetic_ean13_image):
    """白名单不含目标格式时返回空。"""
    img = read_image(synthetic_ean13_image)
    results = decode_region(img, formats=["QRCode"])
    assert results == []


def test_decode_unknown_format_raises(synthetic_ean13_image):
    """未知格式名应抛 ValueError 且消息列出可用格式。"""
    img = read_image(synthetic_ean13_image)
    import pytest
    with pytest.raises(ValueError, match="Unknown barcode format"):
        decode_region(img, formats=["NOT_A_FORMAT"])


def test_decoded_position_is_four_corners(synthetic_ean13_image):
    """position 始终是 4 个 (int, int) 角点。"""
    img = read_image(synthetic_ean13_image)
    results = decode_region(img)
    assert len(results) >= 1
    pos = results[0].position
    assert len(pos) == 4
    for corner in pos:
        assert isinstance(corner, tuple) and len(corner) == 2
        assert all(isinstance(c, int) for c in corner)
