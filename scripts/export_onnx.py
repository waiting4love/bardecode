"""一次性脚本：下载 Piero2411/YOLOV8s-Barcode-Detection.pt 并导出 ONNX。

运行方式（不污染项目环境）：
    uv run --with ultralytics --with requests python scripts/export_onnx.py
"""
import requests
from pathlib import Path
from ultralytics import YOLO

PT_URL = "https://huggingface.co/Piero2411/YOLOV8s-Barcode-Detection/resolve/main/YOLOV8s_Barcode_Detection.pt"
OUTPUT_DIR = Path(__file__).parent.parent / "bardecode" / "models"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = OUTPUT_DIR / "yolov8s_barcode.pt"
    onnx_path = OUTPUT_DIR / "yolov8s_barcode.onnx"

    if not pt_path.exists():
        print(f"Downloading {PT_URL} ...")
        r = requests.get(PT_URL, timeout=180)
        r.raise_for_status()
        pt_path.write_bytes(r.content)
        print(f"Saved {pt_path} ({pt_path.stat().st_size / 1e6:.1f} MB)")

    print("Loading model and exporting ONNX ...")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=640, simplify=True, dynamic=False, opset=12)

    # ultralytics returns the export path as a string; rename to target if needed
    exported_path = Path(exported)
    if exported_path != onnx_path:
        if onnx_path.exists():
            onnx_path.unlink()
        exported_path.rename(onnx_path)

    # Delete intermediate .pt (not packaged into wheel, gitignored)
    if pt_path.exists():
        pt_path.unlink()

    print(f"Done: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
