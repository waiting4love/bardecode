import cv2
import numpy as np
import onnxruntime as ort
from importlib import resources

from .schemas import Detection


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


def postprocess(raw_output: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> list[Detection]:
    """对 YOLOv8 原始输出 [1, 4+nc, N] 做置信度过滤 + NMS，返回 xyxy Detection 列表。

    坐标系：输入空间（640x640）。Detection 为 xyxy 格式。
    nc-agnostic：跨所有类别取最大分数。
    """
    pred = raw_output[0]  # (4+nc, N)
    boxes_cxcywh = pred[:4, :].T  # (N, 4)
    scores = pred[4:, :].max(axis=0)  # (N,) max across classes

    mask = scores >= conf_thres
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return []

    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)
    # cv2.dnn.NMSBoxes expects [x, y, w, h] with (x, y) = top-left corner;
    # passing cxcywh would miscompute IoU for boxes of differing sizes.
    boxes_xywh_tl = np.stack(
        [boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]], axis=1
    )
    idx = cv2.dnn.NMSBoxes(boxes_xywh_tl.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []
    idx = np.array(idx).flatten()

    out = []
    for i in idx:
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        out.append(Detection(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), score=float(scores[i])))
    return out


def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _default_model_path() -> str:
    return str(resources.files("bardecode").joinpath("models", "yolov8s_barcode.onnx"))


class BarcodeDetector:
    def __init__(self, model_path: str | None = None, img_size: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.45):
        path = model_path or _default_model_path()
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, img: np.ndarray) -> list[Detection]:
        """img: BGR numpy array. Returns Detection list in ORIGINAL image coordinates."""
        boxed, ratio, (pad_w, pad_h) = letterbox(img, self.img_size)
        blob = self._preprocess(boxed)
        raw = self.session.run(None, {self.input_name: blob})[0]
        dets_640 = postprocess(raw, self.conf_thres, self.iou_thres)
        h, w = img.shape[:2]
        out = []
        for d in dets_640:
            x1 = int((d.x1 - pad_w) * ratio)
            y1 = int((d.y1 - pad_h) * ratio)
            x2 = int((d.x2 - pad_w) * ratio)
            y2 = int((d.y2 - pad_h) * ratio)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=d.score))
        return out

    @staticmethod
    def _preprocess(boxed: np.ndarray) -> np.ndarray:
        rgb = boxed[:, :, ::-1]
        norm = rgb.astype(np.float32) / 255.0
        nchw = norm.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(nchw)
