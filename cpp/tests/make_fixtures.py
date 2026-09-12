"""Generate test fixture images once, offline (mirrors tests/conftest.py).

Run from repo root with dev deps available:
    uv run python cpp/tests/make_fixtures.py
"""

from pathlib import Path

import cv2
import numpy as np
from barcode import EAN13
from barcode.writer import ImageWriter

FIXTURES = Path(__file__).parent / "fixtures"


def make_ean13_png(path: Path, value: str = "4006381333931") -> None:
    value = EAN13(value, writer=ImageWriter())
    value.save(str(path.with_suffix("")), options={"module_height": 18, "write_text": False})
    # python-barcode writes PNG; ensure grayscale -> BGR color for consistency
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(str(path), img)


def make_ean13_on_photo(path: Path) -> None:
    """EAN-13 pasted onto a noisy 800x600 'photo' background."""
    barcode_path = path.with_suffix(".tmp.png")
    make_ean13_png(barcode_path)
    bc = cv2.imread(str(barcode_path), cv2.IMREAD_COLOR)
    h, w = bc.shape[:2]
    rng = np.random.default_rng(42)
    photo = rng.integers(90, 160, size=(600, 800, 3), dtype=np.uint8)
    y0, x0 = (600 - h) // 2, (800 - w) // 2
    photo[y0:y0 + h, x0:x0 + w] = bc
    cv2.imwrite(str(path), photo)
    barcode_path.unlink()


def make_blank_png(path: Path) -> None:
    cv2.imwrite(str(path), np.full((200, 300, 3), 128, dtype=np.uint8))


def make_qr_png(path: Path) -> None:
    data = "https://example.com/bardecode"
    enc = cv2.QRCodeEncoder.create()
    qr = enc.encode(data)
    qr = cv2.resize(qr, (qr.shape[0] * 8, qr.shape[1] * 8), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), qr)


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    make_ean13_png(FIXTURES / "ean13.png")
    make_ean13_on_photo(FIXTURES / "ean13_photo.png")
    make_blank_png(FIXTURES / "blank.png")
    make_qr_png(FIXTURES / "qr.png")
    print(f"fixtures written to {FIXTURES}")
