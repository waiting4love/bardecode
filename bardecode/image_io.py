import json

import cv2
import numpy as np
from PIL import Image

from .schemas import Report, ImageResult


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = _read_via_pillow(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _read_via_pillow(path: str) -> np.ndarray | None:
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)[:, :, ::-1].copy()
        return arr
    except Exception:
        return None


def format_report_json(report: Report, quiet: bool = False) -> str:
    if quiet:
        lines = []
        for ir in report.results:
            for b in ir.barcodes:
                lines.append(b.text)
        return "\n".join(lines)
    return json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2)


def _report_to_dict(report: Report) -> dict:
    return {
        "tool": report.tool,
        "version": report.version,
        "results": [_image_result_to_dict(ir) for ir in report.results],
    }


def _image_result_to_dict(ir: ImageResult) -> dict:
    return {
        "image": ir.image,
        "width": ir.width,
        "height": ir.height,
        "barcodes": [
            {
                "format": b.format,
                "text": b.text,
                "bbox": list(b.bbox),
                "detection_score": b.detection_score,
            }
            for b in ir.barcodes
        ],
        "undecoded": list(ir.undecoded),
        "error": ir.error,
    }
