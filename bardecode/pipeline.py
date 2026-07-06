from .schemas import DecodedBarcode, ImageResult
from .image_io import read_image
from .decoder import decode_region


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    return inter / (aarea + barea - inter)


_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from .detector import BarcodeDetector
        _detector = BarcodeDetector()
    return _detector


def process_image(
    path: str,
    formats: list[str] | None = None,
    fallback: bool = True,
    detector: "BarcodeDetector | None" = None,
) -> ImageResult:
    try:
        img = read_image(path)
    except FileNotFoundError:
        return ImageResult(image=path, width=0, height=0, barcodes=[], undecoded=[], error="file not found")
    except Exception as e:
        return ImageResult(image=path, width=0, height=0, barcodes=[], undecoded=[], error=f"read error: {e}")

    h, w = img.shape[:2]
    try:
        det = detector or _get_detector()
        detections = det.detect(img)
    except Exception as e:
        return ImageResult(image=path, width=w, height=h, error=f"detect error: {e}")

    barcodes = []
    undecoded = []
    for d in detections:
        crop = img[d.y1:d.y2, d.x1:d.x2]
        if crop.size == 0:
            continue
        try:
            results = decode_region(crop, formats=formats)
        except Exception:
            results = []
        if results:
            for r in results:
                barcodes.append(DecodedBarcode(
                    format=r.format,
                    text=r.text,
                    bbox=[d.x1, d.y1, d.x2, d.y2],
                    detection_score=d.score,
                ))
        else:
            undecoded.append({"bbox": [d.x1, d.y1, d.x2, d.y2], "detection_score": d.score})

    if fallback and not barcodes:
        try:
            full_results = decode_region(img, formats=formats)
        except Exception:
            full_results = []
        for r in full_results:
            xs = [p[0] for p in r.position]
            ys = [p[1] for p in r.position]
            fb_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            barcodes.append(DecodedBarcode(
                format=r.format,
                text=r.text,
                bbox=[fb_bbox[0], fb_bbox[1], fb_bbox[2], fb_bbox[3]],
                detection_score=0.0,
            ))
            # Remove overlapping undecoded entries (we now have the decoded version)
            undecoded = [u for u in undecoded if _iou(tuple(u["bbox"]), fb_bbox) <= 0.5]

    seen = set()
    deduped = []
    for b in barcodes:
        key = (b.format, b.text, tuple(b.bbox))
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    barcodes = deduped

    return ImageResult(
        image=path, width=w, height=h,
        barcodes=barcodes, undecoded=undecoded, error=None,
    )
