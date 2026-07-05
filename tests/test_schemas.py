from bardecode.schemas import (
    Detection, DecodedBarcode, ImageResult, Report
)


def test_detection_fields():
    d = Detection(x1=10, y1=20, x2=30, y2=40, score=0.9)
    assert d.x1 == 10 and d.y2 == 40 and d.score == 0.9


def test_decoded_barcode_fields():
    b = DecodedBarcode(
        format="EAN_13",
        text="4006381333918",
        bbox=[10, 20, 30, 40],
        detection_score=0.9,
    )
    assert b.format == "EAN_13"
    assert b.text == "4006381333918"
    assert b.bbox == [10, 20, 30, 40]


def test_image_result_with_error():
    r = ImageResult(image="x.jpg", width=0, height=0, barcodes=[], undecoded=[], error="missing")
    assert r.error == "missing"
    assert r.barcodes == []


def test_report_round_trip():
    r = Report(tool="bardecode", version="0.1.0", results=[])
    assert r.tool == "bardecode"
    assert r.results == []


def test_image_result_defaults_are_independent():
    a = ImageResult(image="a", width=0, height=0)
    b = ImageResult(image="b", width=0, height=0)
    a.barcodes.append(DecodedBarcode("X", "1", [0, 0, 0, 0], 1.0))
    a.undecoded.append({"bbox": [0, 0, 1, 1], "detection_score": 0.5})
    # Two instances must not share mutable defaults
    assert b.barcodes == []
    assert b.undecoded == []


def test_report_defaults_are_independent():
    a = Report(tool="bardecode", version="0.1.0")
    b = Report(tool="bardecode", version="0.1.0")
    a.results.append(ImageResult(image="x", width=0, height=0))
    assert b.results == []
