from dataclasses import dataclass
import numpy as np
import zxingcpp
from zxingcpp import BarcodeFormat, BarcodeFormats


@dataclass
class Decoded:
    format: str
    text: str
    position: list[tuple[int, int]]


def decode_region(image: np.ndarray, formats: list[str] | None = None) -> list[Decoded]:
    """对一张 numpy 图（BGR 或灰度）解码所有可见条形码。

    formats: 可选白名单，元素须为 zxing-cpp BarcodeFormat 枚举名
             （如 "EAN13"、"QRCode"、"DataMatrix"、"Code128"）。
             传 None 表示接受所有格式。
    """
    kwargs = {}
    if formats:
        kwargs["formats"] = _resolve_formats(formats)
    results = zxingcpp.read_barcodes(image, **kwargs)
    out = []
    for r in results:
        pos = [
            (int(r.position.top_left.x), int(r.position.top_left.y)),
            (int(r.position.top_right.x), int(r.position.top_right.y)),
            (int(r.position.bottom_right.x), int(r.position.bottom_right.y)),
            (int(r.position.bottom_left.x), int(r.position.bottom_left.y)),
        ]
        out.append(Decoded(format=str(r.format), text=r.text, position=pos))
    return out


def _resolve_formats(formats: list[str]) -> BarcodeFormats:
    members = BarcodeFormat.__members__
    resolved = []
    for f in formats:
        name = f.strip()
        try:
            resolved.append(members[name])
        except KeyError:
            valid = list(members)
            raise ValueError(f"Unknown barcode format: {name!r}. Valid: {valid}")
    return BarcodeFormats(resolved)
