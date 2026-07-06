import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    target: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """按保持比例缩放 + 灰边填充到 target x target。

    Returns:
        (out, ratio, (pad_w, pad_h))
        - out: target x target x C uint8 array (same channel count as img)
        - ratio = src_size / dst_size (original_coord = (padded_coord - pad) * ratio)
        - pad_w, pad_h: left/top padding pixels in the output space
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2

    out = np.full((target, target, img.shape[2]), color, dtype=np.uint8)
    out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    ratio = 1.0 / scale  # src/dst
    return out, ratio, (pad_w, pad_h)
