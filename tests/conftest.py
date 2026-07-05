import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def blank_image(tmp_path):
    """300x200 纯灰图，无条形码。"""
    arr = np.full((200, 300, 3), 128, dtype=np.uint8)
    p = tmp_path / "blank.png"
    Image.fromarray(arr).save(p)
    return str(p)


@pytest.fixture
def png_image(tmp_path):
    """带简单图案的 PNG，用于测试 cv2 能读。"""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[40:60, 10:90] = 255
    p = tmp_path / "test.png"
    Image.fromarray(arr).save(p)
    return str(p)


@pytest.fixture
def webp_image(tmp_path):
    """webp 格式图。"""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[40:60, 10:90] = 255
    p = tmp_path / "test.webp"
    Image.fromarray(arr).save(p, format="WEBP")
    return str(p)


@pytest.fixture
def synthetic_ean13_image(tmp_path):
    """生成含 EAN-13 条码的合成图。"""
    import barcode
    from barcode.writer import ImageWriter
    ean = barcode.get("ean13", "4006381333917", writer=ImageWriter())
    fn = ean.save(str(tmp_path / "ean13"))
    return fn
