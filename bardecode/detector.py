import cv2
import numpy as np


def letterbox(img: np.ndarray, target: int = 640, color=(114, 114, 114)):
    """按保持比例缩放 + 灰边填充到 target x target。

    Returns:
        (out, ratio, (pad_w, pad_h))
        - out: target x target x C uint8 array
        - ratio = src_size / dst_size (multiply dst coords by ratio to get src coords)
        - pad_w, pad_h: left/top padding pixels in the output space
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2

    channels = img.shape[2] if img.ndim == 3 else 1
    out = np.full((target, target, channels), color, dtype=np.uint8)
    if img.ndim == 3:
        out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    else:
        out[pad_h:pad_h + new_h, pad_w:pad_w + new_w, 0] = resized
    ratio = 1.0 / scale  # src/dst
    return out, ratio, (pad_w, pad_h)
