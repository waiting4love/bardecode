import pytest
from bardecode.pipeline import process_image, _iou


def test_iou_identical():
    assert abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6


def test_iou_disjoint():
    assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_iou_half_overlap():
    # (0,0,10,10) area=100; (5,0,15,10) area=100; intersection=5*10=50; union=150
    assert abs(_iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_pipeline_missing_file(tmp_path):
    result = process_image(str(tmp_path / "nope.png"))
    assert result.error == "file not found"
    assert result.barcodes == []


@pytest.mark.slow
def test_pipeline_synthetic_ean13(synthetic_ean13_image):
    result = process_image(synthetic_ean13_image)
    assert result.error is None
    assert len(result.barcodes) >= 1
    assert any("4006381333917" in b.text for b in result.barcodes)


@pytest.mark.slow
def test_pipeline_blank(blank_image):
    result = process_image(blank_image)
    assert result.error is None
    assert result.barcodes == []
