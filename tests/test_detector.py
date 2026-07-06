import numpy as np
from bardecode.detector import letterbox, postprocess


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


def test_letterbox_round_trip_coordinate():
    """Verify the inverse mapping contract: original = (padded - pad) * ratio."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    # Mark a known source block centered at (x=100, y=50). A small solid block
    # is used instead of a single pixel so interior pixels stay saturated (255)
    # after INTER_LINEAR resize; its centroid still represents (100, 50).
    img[48:53, 98:103] = [255, 0, 0]
    out, ratio, (pad_w, pad_h) = letterbox(img, 640)
    # Centroid of saturated red pixels in the output
    ys, xs = np.where(out[:, :, 0] == 255)
    assert len(xs) > 0
    out_x = float(xs.mean())
    out_y = float(ys.mean())
    # Inverse transform should recover the original coordinate within 1 pixel
    recon_x = (out_x - pad_w) * ratio
    recon_y = (out_y - pad_h) * ratio
    assert abs(recon_x - 100) <= 1
    assert abs(recon_y - 50) <= 1


def _fake_raw_output_nc2():
    """模拟 nc=2 的 YOLOv8 输出 [1, 6, 3]：3 个候选，4 bbox + 2 类分数。

    候选 0：左上，高置信度（class0=0.95, class1=0.10）
    候选 1：右下，低置信度（应被过滤，max=0.10 < 0.25）
    候选 2：与 0 高度重叠，中置信度（class0=0.85）→ 应被 NMS 抑制
    """
    arr = np.array([
        [[100, 300, 105],   # cx
         [100, 300, 102],   # cy
         [100, 50,  98],    # w
         [100, 50,  99],    # h
         [0.95, 0.10, 0.85],  # class 0 score
         [0.05, 0.03, 0.02]]  # class 1 score
    ], dtype=np.float32)  # shape (1, 6, 3)
    return arr


def test_postprocess_filters_low_conf():
    raw = _fake_raw_output_nc2()
    dets = postprocess(raw, conf_thres=0.25, iou_thres=0.45)
    assert all(d.score >= 0.25 for d in dets)


def test_postprocess_nms_suppresses_duplicate():
    raw = _fake_raw_output_nc2()
    dets = postprocess(raw, conf_thres=0.25, iou_thres=0.45)
    # 候选 0 和 2 几乎相同位置，候选 1 低置信度 → 只剩 1 个
    assert len(dets) == 1


def test_postprocess_returns_xyxy_detection():
    raw = _fake_raw_output_nc2()
    dets = postprocess(raw, conf_thres=0.25, iou_thres=0.45)
    d = dets[0]
    # xyxy semantics: x1 < x2, y1 < y2
    assert d.x1 < d.x2
    assert d.y1 < d.y2
    assert abs(d.score - 0.95) < 1e-5  # highest conf candidate kept (float32 tol)


def test_postprocess_empty_when_all_filtered():
    raw = _fake_raw_output_nc2()
    dets = postprocess(raw, conf_thres=0.99, iou_thres=0.45)
    assert dets == []


def test_postprocess_nms_uses_top_left_xywh():
    """回归：NMS 必须用左上角 xywh 计算 IoU，而非 cxcywh。

    cv2.dnn.NMSBoxes 把输入当作 [x,y,w,h](左上角)。两个候选在 IoU 阈值附近时，
    只有按正确几何（左上角）计算 IoU 才会抑制重复。若误传 cxcywh，IoU 被算低，
    本应被抑制的候选会保留。
    Box A: cxcywh(100,100,100,100)  Box B: cxcywh(140,100,120,100)
    正确 IoU(xywh)=7000/15000≈0.467 > 0.45 -> 抑制 B
    错误 IoU(cxcywh)=6000/16000=0.375 < 0.45 -> 保留 B（bug）
    """
    arr = np.array([
        [[100, 140],   # cx
         [100, 100],   # cy
         [100, 120],   # w
         [100, 100],   # h
         [0.90, 0.85],  # class 0
         [0.01, 0.02]]  # class 1
    ], dtype=np.float32)  # shape (1, 6, 2)
    dets = postprocess(arr, conf_thres=0.25, iou_thres=0.45)
    assert len(dets) == 1  # B 应被正确抑制
    assert abs(dets[0].score - 0.90) < 1e-5
