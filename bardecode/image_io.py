import cv2
import numpy as np
from PIL import Image


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
