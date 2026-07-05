import json

import pytest
from bardecode.image_io import format_report_json, read_image
from bardecode.schemas import (
    DecodedBarcode, ImageResult, Report
)


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


def test_read_image_uses_pillow_fallback(monkeypatch, png_image):
    """When cv2.imread returns None, the Pillow fallback is used."""
    import cv2 as _cv2
    monkeypatch.setattr(_cv2, "imread", lambda *a, **k: None)
    img = read_image(png_image)
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_json_empty_report():
    report = Report(tool="bardecode", version="0.1.0", results=[])
    s = format_report_json(report)
    data = json.loads(s)
    assert data["tool"] == "bardecode"
    assert data["results"] == []


def test_json_with_barcode():
    ir = ImageResult(
        image="x.jpg", width=100, height=200,
        barcodes=[DecodedBarcode("EAN_13", "123", [1, 2, 3, 4], 0.9)],
        undecoded=[{"bbox": [5, 6, 7, 8], "detection_score": 0.4}],
        error=None,
    )
    report = Report(tool="bardecode", version="0.1.0", results=[ir])
    data = json.loads(format_report_json(report))
    assert data["results"][0]["barcodes"][0]["text"] == "123"
    assert data["results"][0]["undecoded"][0]["detection_score"] == 0.4
    assert data["results"][0]["error"] is None


def test_json_quiet():
    ir = ImageResult(
        image="x.jpg", width=100, height=200,
        barcodes=[
            DecodedBarcode("EAN_13", "111", [1, 2, 3, 4], 0.9),
            DecodedBarcode("QR_CODE", "222", [5, 6, 7, 8], 0.8),
        ],
    )
    report = Report(tool="bardecode", version="0.1.0", results=[ir])
    s = format_report_json(report, quiet=True)
    assert s.strip() == "111\n222"
