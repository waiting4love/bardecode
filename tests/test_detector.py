import numpy as np
from bardecode.detector import letterbox


def test_letterbox_no_resize_needed():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    out, ratio, (pad_w, pad_h) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert ratio == 1.0
    assert pad_w == 0 and pad_h == 0


def test_letterbox_upscale_square():
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    out, ratio, (pad_w, pad_h) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    # ratio = src/dst = 320/640 = 0.5
    assert abs(ratio - 0.5) < 1e-6
    assert pad_w == 0 and pad_h == 0


def test_letterbox_nonsquare_pads_gray():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:] = 50
    out, ratio, (pad_w, pad_h) = letterbox(img, 640, color=(114, 114, 114))
    assert out.shape == (640, 640, 3)
    assert out[0, 0, 0] == 114  # top padding row is gray
    cy = out.shape[0] // 2
    cx = out.shape[1] // 2
    assert out[cy, cx, 0] == 50  # content preserved


def test_letterbox_ratio_inverse():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    _, ratio, (pad_w, pad_h) = letterbox(img, 640)
    # ratio = src_size / dst_size; for 400-wide scaled to fit 640: scale=640/400=1.6
    assert abs(ratio - (400 / 640)) < 1e-3
    assert pad_w == 0
    assert abs(pad_h - 160) < 5
