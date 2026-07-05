from dataclasses import dataclass
from typing import Optional
import zxingcpp


@dataclass
class Decoded:
    format: str
    text: str
    position: list[tuple[int, int]]


def decode_region(image, formats: Optional[list[str]] = None) -> list[Decoded]:
    """对一张 numpy 图（BGR 或灰度）解码所有可见条码。"""
    kwargs = {}
    if formats:
        kwargs["formats"] = [str(f) for f in formats]
    results = zxingcpp.read_barcodes(image, **kwargs)
    out = []
    for r in results:
        p = r.position
        pos = [
            (int(p.top_left.x), int(p.top_left.y)),
            (int(p.top_right.x), int(p.top_right.y)),
            (int(p.bottom_right.x), int(p.bottom_right.y)),
            (int(p.bottom_left.x), int(p.bottom_left.y)),
        ]
        out.append(Decoded(format=str(r.format), text=r.text, position=pos))
    return out
