# bardecode 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个 CLI 工具 `bardecode`，从照片中检测并解码所有 1D/2D 条形码，运行在 CPU x86 小主机上。

**Architecture:** YOLOv8s-barcode ONNX 检测（onnxruntime CPU）→ 裁剪 → zxing-cpp 解码，纯 Python，依赖最小化。

**Tech Stack:** Python 3.10+、uv、onnxruntime、zxing-cpp 3.x、opencv-python-headless、numpy、pytest。

---

## 对 spec 的偏离说明（已与用户讨论确认）

| spec 章节 | 原设计 | 实际实现 | 原因 |
|---|---|---|---|
| 5.2 权重源 | 首次自动下载 ONNX 到 `~/.cache/` | 开发期导出 ONNX 后**打包进 wheel**，运行时从包内加载 | 真实世界没有现成的开源 YOLOv8n-barcode ONNX；最热门权重 `Piero2411/YOLOV8s-Barcode-Detection` 只有 `.pt`，需 PyTorch 导出。打包进 wheel 让弱主机开箱即用、零网络依赖 |
| 5.2 兜底 | 用 `BARDECODE_MODEL_URL` 自定义 | 保留 `--model PATH` 覆盖；移除环境变量与 `download-model` 子命令 | YAGNI：包内已有 ONNX，无需下载逻辑 |
| 5.1 模型 | YOLOv8**n** (~3MB) | YOLOv8**s** (~22MB ONNX) | 仅 YOLOv8s 的 barcode 微调权重是公开且经过验证的（1529 下载）。22MB 在 8GB+ 内存上完全可接受 |

其他章节保持不变。

---

## 文件结构

```
bardecode/
├── __init__.py              # 版本号
├── __main__.py              # CLI 入口 (argparse)，<120 行
├── pipeline.py              # 编排：读图→检测→裁剪→解码→汇总
├── detector.py              # YOLOv8 ONNX 检测器 + 后处理
├── decoder.py               # zxing-cpp 封装
├── image_io.py              # 读图（webp 兜底）+ JSON 格式化
├── schemas.py               # dataclass
└── models/
    └── yolov8s_barcode.onnx  # 打包进 wheel 的预训练权重（22MB）
scripts/
└── export_onnx.py           # 一次性脚本：下载 .pt → 导出 ONNX
tests/
├── conftest.py              # 共享 fixture：合成条码图、负样本
├── test_schemas.py
├── test_image_io.py
├── test_decoder.py
├── test_detector.py
├── test_pipeline.py
└── test_cli.py
pyproject.toml
README.md
.gitignore
```

每个 `.py` 文件单一职责，便于测试与上下文管理。

---

## Task 1: 项目骨架与依赖

**Files:**
- Create: `pyproject.toml`
- Create: `bardecode/__init__.py`
- Create: `.gitignore`
- Create: `tests/__init__.py`（空）
- Create: `tests/conftest.py`（占位，后续任务填充）

- [ ] **Step 1: 用 uv 初始化项目**

在 PowerShell 中执行（工作目录 `D:\Temp\opencv-barcode`）：

```powershell
uv init --name bardecode --package --python 3.11
```

如果 `uv init` 生成了不需要的模板文件（如 `hello.py`、`README.md` 默认内容），手动删除 `hello.py`。检查生成的 `pyproject.toml`。

- [ ] **Step 2: 覆写 pyproject.toml**

把 `pyproject.toml` 内容完全替换为：

```toml
[project]
name = "bardecode"
version = "0.1.0"
description = "Detect and decode 1D/2D barcodes from photos using a lightweight YOLOv8 ONNX model"
requires-python = ">=3.10"
dependencies = [
    "onnxruntime>=1.16",
    "zxing-cpp>=2.0",
    "opencv-python-headless>=4.8",
    "numpy>=1.24",
]

[project.scripts]
bardecode = "bardecode.__main__:main"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "python-barcode>=0.14",
    "Pillow>=10.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["bardecode"]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 3: 创建 bardecode/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: 创建 .gitignore**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
*.egg-info/
dist/
build/
models/*.pt
models/*.onnx.tmp
```

注意：`models/*.onnx` **不**忽略（要打包进 wheel）。

- [ ] **Step 5: 创建空 tests/__init__.py 和占位 conftest.py**

`tests/__init__.py`：空文件。

`tests/conftest.py`：
```python
# 后续任务会在此添加共享 fixture
```

- [ ] **Step 6: 安装依赖并验证**

```powershell
uv sync --extra dev
uv run pytest --version
```

预期：pytest 打印版本号，无错误。

- [ ] **Step 7: 初始化 git 并首次提交**

```powershell
git init
git add .
git commit -m "chore: project skeleton with uv"
```

---

## Task 2: schemas.py — 数据类

**Files:**
- Create: `bardecode/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: 写失败测试 `tests/test_schemas.py`**

```python
from bardecode.schemas import (
    Detection, DecodedBarcode, ImageResult, Report
)


def test_detection_fields():
    d = Detection(x1=10, y1=20, x2=30, y2=40, score=0.9)
    assert d.x1 == 10 and d.y2 == 40 and d.score == 0.9


def test_decoded_barcode_fields():
    b = DecodedBarcode(
        format="EAN_13",
        text="4006381333918",
        bbox=[10, 20, 30, 40],
        detection_score=0.9,
    )
    assert b.format == "EAN_13"
    assert b.text == "4006381333918"
    assert b.bbox == [10, 20, 30, 40]


def test_image_result_with_error():
    r = ImageResult(image="x.jpg", width=0, height=0, barcodes=[], undecoded=[], error="missing")
    assert r.error == "missing"
    assert r.barcodes == []


def test_report_round_trip():
    r = Report(tool="bardecode", version="0.1.0", results=[])
    assert r.tool == "bardecode"
    assert r.results == []
```

- [ ] **Step 2: 验证失败**

```powershell
uv run pytest tests/test_schemas.py -v
```

预期：`ModuleNotFoundError: No module named 'bardecode.schemas'`。

- [ ] **Step 3: 实现 `bardecode/schemas.py`**

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass
class DecodedBarcode:
    format: str
    text: str
    bbox: list[int]
    detection_score: float


@dataclass
class ImageResult:
    image: str
    width: int
    height: int
    barcodes: list[DecodedBarcode] = field(default_factory=list)
    undecoded: list[dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Report:
    tool: str
    version: str
    results: list[ImageResult] = field(default_factory=list)
```

- [ ] **Step 4: 验证测试通过**

```powershell
uv run pytest tests/test_schemas.py -v
```

预期：4 passed。

- [ ] **Step 5: 提交**

```powershell
git add bardecode/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add dataclasses for detection results"
```

---

## Task 3: image_io.py — 读图（含 webp 兜底）

**Files:**
- Modify: `bardecode/image_io.py`（创建）
- Test: `tests/test_image_io.py`
- Modify: `tests/conftest.py`（添加合成图 fixture）

- [ ] **Step 1: 在 conftest.py 添加共享 fixture**

```python
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def blank_image(tmp_path):
    """300x200 纯灰图，无条形码。"""
    arr = np.full((200, 300, 3), 128, dtype=np.uint8)
    p = tmp_path / "blank.png"
    Image.fromarray(arr).save(p)
    return str(p)


@pytest.fixture
def png_image(tmp_path):
    """带简单图案的 PNG，用于测试 cv2 能读。"""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[40:60, 10:90] = 255
    p = tmp_path / "test.png"
    Image.fromarray(arr).save(p)
    return str(p)


@pytest.fixture
def webp_image(tmp_path):
    """webp 格式图。"""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[40:60, 10:90] = 255
    p = tmp_path / "test.webp"
    Image.fromarray(arr).save(p, format="WEBP")
    return str(p)
```

- [ ] **Step 2: 写失败测试 `tests/test_image_io.py`**

```python
import pytest
from bardecode.image_io import read_image


def test_read_png(png_image):
    img = read_image(png_image)
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_read_webp(webp_image):
    img = read_image(webp_image)
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_read_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(str(tmp_path / "nope.png"))
```

- [ ] **Step 3: 验证失败**

```powershell
uv run pytest tests/test_image_io.py -v
```

预期：`ModuleNotFoundError`。

- [ ] **Step 4: 实现 `bardecode/image_io.py`**

```python
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


def _read_via_pillow(path: str):
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)[:, :, ::-1].copy()
        return arr
    except Exception:
        return None
```

- [ ] **Step 5: 验证测试通过**

```powershell
uv run pytest tests/test_image_io.py -v
```

预期：3 passed。

- [ ] **Step 6: 提交**

```powershell
git add bardecode/image_io.py tests/test_image_io.py tests/conftest.py
git commit -m "feat(image_io): read image with webp fallback via Pillow"
```

---

## Task 4: image_io.py — JSON 格式化

**Files:**
- Modify: `bardecode/image_io.py`
- Test: `tests/test_image_io.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_image_io.py`）**

```python
import json
from bardecode.image_io import format_report_json
from bardecode.schemas import (
    DecodedBarcode, ImageResult, Report
)


def test_json_empty_report():
    report = Report(tool="bardecode", version="0.1.0", results=[])
    s = format_report_json(report)
    data = json.loads(s)
    assert data["tool"] == "bardecode"
    assert data["results"] == []


def test_json_with_barcode():
    ir = ImageResult(
        image="x.jpg", width=100, height=200,
        barcodes=[DecodedBarcode("EAN_13", "123", [1, 2, 3, 4], 0.9)],
        undecoded=[{"bbox": [5, 6, 7, 8], "detection_score": 0.4}],
        error=None,
    )
    report = Report(tool="bardecode", version="0.1.0", results=[ir])
    data = json.loads(format_report_json(report))
    assert data["results"][0]["barcodes"][0]["text"] == "123"
    assert data["results"][0]["undecoded"][0]["detection_score"] == 0.4
    assert data["results"][0]["error"] is None


def test_json_quiet():
    ir = ImageResult(
        image="x.jpg", width=100, height=200,
        barcodes=[
            DecodedBarcode("EAN_13", "111", [1, 2, 3, 4], 0.9),
            DecodedBarcode("QR_CODE", "222", [5, 6, 7, 8], 0.8),
        ],
    )
    report = Report(tool="bardecode", version="0.1.0", results=[ir])
    s = format_report_json(report, quiet=True)
    assert s.strip() == "111\n222"
```

- [ ] **Step 2: 验证失败**

```powershell
uv run pytest tests/test_image_io.py::test_json_empty_report -v
```

预期：`ImportError: cannot import name 'format_report_json'`。

- [ ] **Step 3: 实现 `format_report_json`（追加到 `bardecode/image_io.py`）**

在 `image_io.py` **顶部**确认有相对导入（如果还没有则添加）：

```python
from .schemas import Report
```

然后追加函数：

```python
import json


def format_report_json(report, quiet: bool = False) -> str:
    if quiet:
        lines = []
        for ir in report.results:
            for b in ir.barcodes:
                lines.append(b.text)
        return "\n".join(lines)
    return json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2)


def _report_to_dict(report) -> dict:
    return {
        "tool": report.tool,
        "version": report.version,
        "results": [_image_result_to_dict(ir) for ir in report.results],
    }


def _image_result_to_dict(ir) -> dict:
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
```

- [ ] **Step 4: 验证测试通过**

```powershell
uv run pytest tests/test_image_io.py -v
```

预期：6 passed（原 3 个 + 新增 3 个）。

- [ ] **Step 5: 提交**

```powershell
git add bardecode/image_io.py tests/test_image_io.py
git commit -m "feat(image_io): JSON and quiet output formatting"
```

---

## Task 5: decoder.py — zxing-cpp 封装

**Files:**
- Create: `bardecode/decoder.py`
- Test: `tests/test_decoder.py`
- Modify: `tests/conftest.py`（添加合成条码 fixture）

- [ ] **Step 1: 在 conftest.py 添加合成条码 fixture**

在 conftest.py 末尾追加：

```python
@pytest.fixture
def synthetic_ean13_image(tmp_path):
    """生成含 EAN-13 条码的合成图。"""
    import barcode
    from barcode.writer import ImageWriter
    ean = barcode.get("ean13", "4006381333918", writer=ImageWriter())
    # render() 返回 BytesIO；这里直接让 python-barcode 写到磁盘
    fn = ean.save(str(tmp_path / "ean13"))
    # python-barcode 输出的 PNG 可能有白边；直接读回返回路径
    return fn
```

注意：`python-barcode` 的 `ImageWriter` 需要 Pillow（已在 dev 依赖里）。

- [ ] **Step 2: 写失败测试 `tests/test_decoder.py`**

```python
import cv2
from bardecode.decoder import decode_region
from bardecode.image_io import read_image


def test_decode_synthetic_ean13(synthetic_ean13_image):
    img = read_image(synthetic_ean13_image)
    results = decode_region(img)
    assert len(results) >= 1
    assert any(r.format.upper().startswith("EAN") for r in results)
    assert any("4006381333918" in r.text for r in results)


def test_decode_blank_returns_empty(blank_image):
    img = read_image(blank_image)
    assert decode_region(img) == []
```

- [ ] **Step 3: 验证失败**

```powershell
uv run pytest tests/test_decoder.py -v
```

预期：`ModuleNotFoundError: No module named 'bardecode.decoder'`。

- [ ] **Step 4: 实现 `bardecode/decoder.py`**

```python
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
        pos = [(int(p.x), int(p.y)) for p in r.position]
        out.append(Decoded(format=str(r.format), text=r.text, position=pos))
    return out
```

- [ ] **Step 5: 验证测试通过**

```powershell
uv run pytest tests/test_decoder.py -v
```

预期：2 passed。

如果 `test_decode_synthetic_ean13` 失败（zxing 可能要求更高分辨率），把 fixture 改成放大 2 倍：

```python
@pytest.fixture
def synthetic_ean13_image(tmp_path):
    import barcode
    from barcode.writer import ImageWriter
    import numpy as np
    from PIL import Image
    ean = barcode.get("ean13", "4006381333918", writer=ImageWriter())
    fn = ean.save(str(tmp_path / "ean13"))
    # 放大 2 倍以提高解码成功率
    img = Image.open(fn)
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img.save(fn)
    return fn
```

- [ ] **Step 6: 提交**

```powershell
git add bardecode/decoder.py tests/test_decoder.py tests/conftest.py
git commit -m "feat(decoder): zxing-cpp wrapper with format whitelist"
```

---

## Task 6: 导出 ONNX 模型（一次性）

**Files:**
- Create: `scripts/export_onnx.py`
- Create: `bardecode/models/yolov8s_barcode.onnx`

- [ ] **Step 1: 创建 scripts 目录与脚本**

```powershell
New-Item -ItemType Directory -Path "bardecode\models" -Force | Out-Null
New-Item -ItemType Directory -Path "scripts" -Force | Out-Null
```

`scripts/export_onnx.py`：

```python
"""一次性脚本：下载 Piero2411/YOLOV8s-Barcode-Detection.pt 并导出 ONNX。

运行方式（不污染项目环境）：
    uv run --with ultralytics --with requests python scripts/export_onnx.py
"""
import os
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
        r = requests.get(PT_URL, timeout=120)
        r.raise_for_status()
        pt_path.write_bytes(r.content)

    print("Loading model and exporting ONNX ...")
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=640, simplify=True, dynamic=False, opset=12)

    # ultralytics 会把 onnx 输出到 pt 同目录同 stem
    exported = pt_path.with_suffix(".onnx")
    if exported != onnx_path:
        exported.rename(onnx_path)

    # 删除中间 .pt（不打包进 wheel）
    if pt_path.exists():
        pt_path.unlink()

    print(f"Done: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行导出脚本**

```powershell
uv run --with ultralytics --with requests python scripts/export_onnx.py
```

预期输出末尾：`Done: ...\bardecode\models\yolov8s_barcode.onnx (约 22 MB)`。

如果网络受限无法下载，备用方案：
1. 浏览器手动下载 `https://huggingface.co/Piero2411/YOLOV8s-Barcode-Detection/resolve/main/YOLOV8s_Barcode_Detection.pt` 到 `bardecode/models/yolov8s_barcode.pt`
2. 再执行脚本（脚本会跳过下载直接导出）

- [ ] **Step 3: 验证 ONNX 可加载**

```powershell
uv run python -c "import onnxruntime as ort; s = ort.InferenceSession(r'bardecode\models\yolov8s_barcode.onnx', providers=['CPUExecutionProvider']); print('inputs:', [(i.name, i.shape) for i in s.get_inputs()]); print('outputs:', [(o.name, o.shape) for o in s.get_outputs()])"
```

记录输出张量的实际形状和名字，用于 Task 8 的后处理实现。**预期**：输入 `[1, 3, 640, 640]`；输出类似 `[1, 84, 8400]` 或 `[1, 6, 8400]`（取决于 ultralytics 导出版本）。

- [ ] **Step 4: 提交（注意 .pt 已被 .gitignore 排除）**

```powershell
git add scripts/export_onnx.py bardecode/models/yolov8s_barcode.onnx
git commit -m "chore(model): export YOLOv8s barcode ONNX and bundle in package"
```

注意：22MB 文件 git 提交是可接受的（一次性产物）；后续如需走 Git LFS 再迁移。

---

## Task 7: detector.py — letterbox 函数

**Files:**
- Create: `bardecode/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: 写失败测试 `tests/test_detector.py`**

```python
import numpy as np
import pytest
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
    assert abs(ratio - 2.0) < 1e-6


def test_letterbox_nonsquare_pads_gray():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:] = 50  # 内容色 50
    out, ratio, (pad_w, pad_h) = letterbox(img, 640, color=(114, 114, 114))
    assert out.shape == (640, 640, 3)
    # 检查顶部填充行是 114
    assert out[0, 0, 0] == 114
    # 检查内容区域大致保持
    cy = out.shape[0] // 2
    cx = out.shape[1] // 2
    assert out[cy, cx, 0] == 50


def test_letterbox_ratio_inverse():
    """ratio 是 src/dst，反变换时 dst * ratio = src 坐标。"""
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    _, ratio, (pad_w, pad_h) = letterbox(img, 640)
    # 400 宽放到 640 内：宽比例 640/400=1.6；高 200*1.6=320，pad_h=(640-320)/2=160
    assert abs(ratio - (400 / 640)) < 1e-3  # src/dst
    assert pad_w == 0
    assert abs(pad_h - 160) < 2
```

- [ ] **Step 2: 验证失败**

```powershell
uv run pytest tests/test_detector.py -v
```

预期：`ModuleNotFoundError`。

- [ ] **Step 3: 实现 letterbox（追加到 `bardecode/detector.py`）**

```python
import cv2
import numpy as np


def letterbox(img: np.ndarray, target: int = 640, color=(114, 114, 114)):
    """按保持比例缩放 + 灰边填充到 target x target。

    返回 (out, ratio, (pad_w, pad_h))
        ratio = src_size / dst_size（用于反变换坐标）
        pad_w, pad_h = 左/上填充像素数
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2

    out = np.full((target, target, img.shape[2] if img.ndim == 3 else 1), color, dtype=np.uint8)
    if img.ndim == 3:
        out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    else:
        out[pad_h:pad_h + new_h, pad_w:pad_w + new_w, 0] = resized
    ratio = 1.0 / scale  # src/dst
    return out, ratio, (pad_w, pad_h)
```

- [ ] **Step 4: 验证测试通过**

```powershell
uv run pytest tests/test_detector.py -v
```

预期：4 passed。

如果 `test_letterbox_ratio_inverse` 因为取整误差失败，把断言容差放宽到 `< 5`。

- [ ] **Step 5: 提交**

```powershell
git add bardecode/detector.py tests/test_detector.py
git commit -m "feat(detector): letterbox preprocessing"
```

---

## Task 8: detector.py — 后处理（NMS + 坐标反变换）

**Files:**
- Modify: `bardecode/detector.py`
- Test: `tests/test_detector.py`（追加）

**前提**：Task 6 Step 3 已记录输出张量形状。本任务假设 YOLOv8 标准输出 `[1, 4+nc, N]`（nc=1），需要做转置。如果实际形状不同，按实际调整 `postprocess` 函数。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_detector.py`）**

```python
import numpy as np
from bardecode.detector import postprocess, scale_bbox_back


def _fake_raw_output():
    """模拟 YOLOv8 输出 [1, 5, 3]：3 个候选 (cx,cy,w,h,score)，1 类。"""
    # 候选 1：高置信度，左上
    # 候选 2：低置信度，应该被过滤
    # 候选 3：与 1 高度重叠，应被 NMS 抑制
    arr = np.array([
        [[100, 300, 100, 100, 0.95],   # cx,cy,w,h,score
         [400, 400, 50, 50, 0.10],
         [105, 102, 98, 99, 0.85]]
    ], dtype=np.float32)  # shape (1, 5, 3)
    return arr


def test_postprocess_filters_low_conf():
    raw = _fake_raw_output()
    dets = postprocess(raw, conf_thres=0.25, iou_thres=0.45)
    # 低置信度的候选 2 应被过滤
    assert all(d.score >= 0.25 for d in dets)


def test_postprocess_nms_suppresses_duplicate():
    raw = _fake_raw_output()
    dets = postprocess(raw, conf_thres=0.25, iou_thres=0.45)
    # 候选 1 和 3 几乎相同位置，应只剩 1 个
    assert len(dets) == 1


def test_scale_bbox_back():
    """640 坐标反变换回原图。"""
    # 原图 320x320，ratio=0.5，pad=0
    det = (50, 60, 100, 120, 0.9)  # cx,cy,w,h,score in 640-space
    x1, y1, x2, y2, score = scale_bbox_back(det, ratio=0.5, pad_w=0, pad_h=0)
    assert (x1, y1, x2, y2) == (25, 30, 50, 60)
```

- [ ] **Step 2: 验证失败**

```powershell
uv run pytest tests/test_detector.py::test_postprocess_filters_low_conf -v
```

预期：`ImportError`。

- [ ] **Step 3: 实现 postprocess 与 scale_bbox_back（追加到 `bardecode/detector.py`）**

```python
from .schemas import Detection


def postprocess(raw_output: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> list[Detection]:
    """对 YOLOv8 原始输出做置信度过滤 + NMS，返回 xyxy Detection 列表。

    假设 raw_output 形状 [1, 4+nc, N]，按类别最大置信度过滤后做 NMS。
    坐标系：640 x 640 输入空间，xyxy。
    """
    pred = raw_output[0]  # shape (4+nc, N)
    # pred: 行 0..3 = cx, cy, w, h；行 4..4+nc-1 = 类别分数
    boxes_cxcywh = pred[:4, :].T  # (N, 4)
    scores = pred[4:, :].max(axis=0)  # (N,)

    mask = scores >= conf_thres
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return []

    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)
    # cv2.dnn.NMSBoxes 要 [x, y, w, h] 的 list
    boxes_for_nms = boxes_cxcywh.tolist()
    idx = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []
    idx = np.array(idx).flatten()
    keep_xyxy = boxes_xyxy[idx]
    keep_scores = scores[idx]

    out = []
    for box, s in zip(keep_xyxy, keep_scores):
        x1, y1, x2, y2 = box.tolist()
        out.append(Detection(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), score=float(s)))
    return out


def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def scale_bbox_back(det, ratio: float, pad_w: int, pad_h: int):
    """把 640 空间的 (cx, cy, w, h, score) 反变换回原图坐标。

    ratio = src_size / dst_size（与 letterbox 一致）
    """
    cx, cy, w, h, score = det
    cx = (cx - pad_w) * ratio
    cy = (cy - pad_h) * ratio
    w = w * ratio
    h = h * ratio
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return x1, y1, x2, y2, float(score)
```

- [ ] **Step 4: 验证测试通过**

```powershell
uv run pytest tests/test_detector.py -v
```

预期：7 passed（letterbox 4 + postprocess 3）。

- [ ] **Step 5: 提交**

```powershell
git add bardecode/detector.py tests/test_detector.py
git commit -m "feat(detector): postprocess with NMS and bbox scaling"
```

---

## Task 9: detector.py — 完整 ONNX 推理封装

**Files:**
- Modify: `bardecode/detector.py`
- Test: `tests/test_detector.py`（追加 slow 标记测试）

- [ ] **Step 1: 写端到端测试（追加到 `tests/test_detector.py`）**

```python
import pytest
from bardecode.detector import BarcodeDetector


@pytest.fixture
def detector():
    return BarcodeDetector()


@pytest.mark.slow
def test_detector_finds_barcode_in_synthetic(detector, synthetic_ean13_image):
    from bardecode.image_io import read_image
    img = read_image(synthetic_ean13_image)
    dets = detector.detect(img)
    assert len(dets) >= 1
    assert all(d.score > 0.25 for d in dets)


@pytest.mark.slow
def test_detector_empty_on_blank(detector, blank_image):
    from bardecode.image_io import read_image
    img = read_image(blank_image)
    dets = detector.detect(img)
    assert dets == []
```

- [ ] **Step 2: 验证失败**

```powershell
uv run pytest tests/test_detector.py::test_detector_finds_barcode_in_synthetic -v
```

预期：`ImportError: cannot import name 'BarcodeDetector'`。

- [ ] **Step 3: 实现 BarcodeDetector 类（追加到 `bardecode/detector.py`）**

```python
import onnxruntime as ort
from importlib import resources


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
        """img: BGR numpy array. 返回原图坐标系下的 Detection 列表。"""
        boxed, ratio, (pad_w, pad_h) = letterbox(img, self.img_size)
        blob = self._preprocess(boxed)
        raw = self.session.run(None, {self.input_name: blob})[0]
        dets_640 = postprocess(raw, self.conf_thres, self.iou_thres)
        # dets_640 已是 xyxy 形式；按 ratio/pad 反变换
        out = []
        for d in dets_640:
            x1 = int((d.x1 - pad_w) * ratio)
            y1 = int((d.y1 - pad_h) * ratio)
            x2 = int((d.x2 - pad_w) * ratio)
            y2 = int((d.y2 - pad_h) * ratio)
            # 裁剪到原图范围
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            out.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=d.score))
        return out

    @staticmethod
    def _preprocess(boxed: np.ndarray) -> np.ndarray:
        # BGR -> RGB, /255, HWCN -> NCHW
        rgb = boxed[:, :, ::-1]
        norm = rgb.astype(np.float32) / 255.0
        nchw = norm.transpose(2, 0, 1)[None, ...]
        return nchw
```

- [ ] **Step 4: 运行 slow 测试**

```powershell
uv run pytest tests/test_detector.py -v -m slow
```

预期：2 passed。

如果 `test_detector_finds_barcode_in_synthetic` 失败（检测不到），可能原因：
1. python-barcode 生成的图太小 → 放大 fixture
2. ONNX 输出形状不是 `[1, 5, N]` → 用 Task 6 Step 3 记录的实际形状调整 `postprocess`（注意转置逻辑）

调试命令：
```powershell
uv run python -c "import onnxruntime as ort; s=ort.InferenceSession(r'bardecode/models/yolov8s_barcode.onnx'); print(s.get_outputs()[0].shape)"
```

- [ ] **Step 5: 运行所有非 slow 测试确认无回归**

```powershell
uv run pytest -v -m "not slow"
```

预期：全部 pass。

- [ ] **Step 6: 提交**

```powershell
git add bardecode/detector.py tests/test_detector.py
git commit -m "feat(detector): full ONNX inference pipeline"
```

---

## Task 10: pipeline.py — 编排

**Files:**
- Create: `bardecode/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写测试 `tests/test_pipeline.py`**

```python
import pytest
from bardecode.pipeline import process_image, _iou


def test_iou_identical():
    assert abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6


def test_iou_disjoint():
    assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


@pytest.mark.slow
def test_pipeline_synthetic_ean13(synthetic_ean13_image):
    result = process_image(synthetic_ean13_image)
    assert result.error is None
    assert len(result.barcodes) >= 1
    assert any("4006381333918" in b.text for b in result.barcodes)


@pytest.mark.slow
def test_pipeline_blank(blank_image):
    result = process_image(blank_image)
    assert result.error is None
    assert result.barcodes == []


def test_pipeline_missing_file(tmp_path):
    result = process_image(str(tmp_path / "nope.png"))
    assert result.error == "file not found"
```

- [ ] **Step 2: 验证失败**

```powershell
uv run pytest tests/test_pipeline.py -v -m "not slow"
```

预期：`ModuleNotFoundError`。

- [ ] **Step 3: 实现 `bardecode/pipeline.py`**

```python
from .schemas import Detection, DecodedBarcode, ImageResult
from .image_io import read_image
from .decoder import decode_region


def _iou(a, b) -> float:
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


# 模块级懒加载单例（避免每次调用都重新加载 ONNX）
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from .detector import BarcodeDetector
        _detector = BarcodeDetector()
    return _detector


def process_image(path: str, conf_thres: float = 0.25, iou_thres: float = 0.45,
                  img_size: int = 640, formats=None, fallback: bool = True,
                  detector=None) -> ImageResult:
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
                    format=r.format, text=r.text,
                    bbox=[d.x1, d.y1, d.x2, d.y2],
                    detection_score=d.score,
                ))
        else:
            undecoded.append({"bbox": [d.x1, d.y1, d.x2, d.y2], "detection_score": d.score})

    # 全图回退
    if fallback and not barcodes:
        try:
            full_results = decode_region(img, formats=formats)
        except Exception:
            full_results = []
        for r in full_results:
            # zxing 的 position 是 4 个点，取包围盒作为 bbox
            xs = [p[0] for p in r.position]
            ys = [p[1] for p in r.position]
            fb_bbox = (min(xs), min(ys), max(xs), max(ys))
            # 去重：与已有 undecoded 的 IoU > 0.5 视为同一个，但仍记录解码结果
            barcodes.append(DecodedBarcode(
                format=r.format, text=r.text,
                bbox=[int(fb_bbox[0]), int(fb_bbox[1]), int(fb_bbox[2]), int(fb_bbox[3])],
                detection_score=0.0,  # 全图回退没有检测置信度
            ))

    return ImageResult(image=path, width=w, height=h, barcodes=barcodes,
                       undecoded=undecoded, error=None)
```

- [ ] **Step 4: 验证测试通过**

```powershell
uv run pytest tests/test_pipeline.py -v -m "not slow"
```

预期：3 passed（test_iou_identical, test_iou_disjoint, test_pipeline_missing_file）。

```powershell
uv run pytest tests/test_pipeline.py -v -m slow
```

预期：2 passed（slow）。

- [ ] **Step 5: 提交**

```powershell
git add bardecode/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): orchestrate detect-crop-decode with full-image fallback"
```

---

## Task 11: __main__.py — CLI

**Files:**
- Create: `bardecode/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写测试 `tests/test_cli.py`**

```python
import json
import subprocess
import sys
from pathlib import Path


def _run(args, **kw):
    return subprocess.run(
        [sys.executable, "-m", "bardecode"] + args,
        capture_output=True, text=True, **kw,
    )


def test_cli_help():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "bardecode" in r.stdout.lower()


def test_cli_missing_image_exits_2(tmp_path):
    r = _run([str(tmp_path / "nope.png")])
    assert r.returncode == 2
    data = json.loads(r.stdout)
    assert data["results"][0]["error"] == "file not found"


def test_cli_quiet_outputs_only_values(blank_image):
    r = _run(["--quiet", blank_image])
    assert r.returncode == 0
    assert r.stdout.strip() == ""  # 无码可输出


@pytest.mark.slow
def test_cli_real_image():
    """使用用户提供的真实 webp 图（位于 tests/fixtures/）。"""
    fixture = Path(__file__).parent / "fixtures" / "top-barcodes-in-retail-header.jpg.webp"
    if not fixture.exists():
        import pytest
        pytest.skip("fixture webp not available")
    r = _run([str(fixture)])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    total = sum(len(ir["barcodes"]) for ir in data["results"])
    assert total >= 1, f"expected at least 1 barcode, got {data}"
```

- [ ] **Step 2: 把用户的 webp 复制到测试夹具目录**

```powershell
New-Item -ItemType Directory -Path "tests\fixtures" -Force | Out-Null
Copy-Item "top-barcodes-in-retail-header.jpg.webp" "tests\fixtures\top-barcodes-in-retail-header.jpg.webp"
```

注意：源文件在仓库根，复制后保留原文件不变。

- [ ] **Step 3: 验证失败**

```powershell
uv run pytest tests/test_cli.py -v -m "not slow"
```

预期：找不到 `bardecode.__main__`。

- [ ] **Step 4: 实现 `bardecode/__main__.py`**

```python
import argparse
import sys
from . import __version__
from .pipeline import process_image
from .image_io import format_report_json
from .schemas import Report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bardecode", description="Detect and decode barcodes from photos.")
    p.add_argument("images", nargs="+", help="image file path(s)")
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.45)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--model", type=str, default=None, help="ONNX model path override")
    p.add_argument("--formats", type=str, default=None, help="comma-separated format whitelist")
    p.add_argument("--no-fallback", action="store_true", help="disable full-image decode fallback")
    p.add_argument("-q", "--quiet", action="store_true", help="print only barcode values")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose to stderr")
    p.add_argument("--version", action="version", version=f"bardecode {__version__}")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    formats = [s.strip() for s in args.formats.split(",")] if args.formats else None

    detector = None
    if args.model:
        from .detector import BarcodeDetector
        detector = BarcodeDetector(
            model_path=args.model,
            img_size=args.img_size,
            conf_thres=args.conf_threshold,
            iou_thres=args.iou_threshold,
        )

    results = []
    exit_code = 0
    from .schemas import ImageResult
    for path in args.images:
        try:
            ir = process_image(
                path,
                conf_thres=args.conf_threshold,
                iou_thres=args.iou_threshold,
                img_size=args.img_size,
                formats=formats,
                fallback=not args.no_fallback,
                detector=detector,
            )
        except Exception as e:
            ir = ImageResult(image=path, width=0, height=0, barcodes=[],
                             undecoded=[], error=f"unexpected: {e}")
        results.append(ir)
        if ir.error == "file not found":
            exit_code = max(exit_code, 2)
        elif ir.error:
            exit_code = max(exit_code, 1)

    report = Report(tool="bardecode", version=__version__, results=results)
    out = format_report_json(report, quiet=args.quiet)
    if out:
        print(out)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 验证非 slow 测试通过**

```powershell
uv run pytest tests/test_cli.py -v -m "not slow"
```

预期：3 passed（help, missing_image, quiet）。

- [ ] **Step 6: 运行 slow 真实图测试**

```powershell
uv run pytest tests/test_cli.py -v -m slow
```

预期：`test_cli_real_image` passed，至少检测到 1 个条码。

如果失败（检测不到）：
- 调低置信度：`bardecode --conf-threshold 0.1 <fixture>`
- 检查 undecoded 列表是否非空（说明检测到但解码失败，可能需要调整裁剪 padding）
- 调试时手动跑：`uv run python -m bardecode --verbose tests\fixtures\top-barcodes-in-retail-header.jpg.webp`

- [ ] **Step 7: 全量回归**

```powershell
uv run pytest -v
```

预期：所有测试通过（slow 与非 slow 一起）。

- [ ] **Step 8: 提交**

```powershell
git add bardecode/__main__.py tests/test_cli.py tests/fixtures/
git commit -m "feat(cli): argparse CLI with JSON/quiet output and exit codes"
```

---

## Task 12: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 README.md**

```markdown
# bardecode

命令行工具：从照片中检测并解码所有 1D/2D 条形码。CPU 推理，针对 x86 小主机优化。

## 安装

\`\`\`powershell
uv sync --extra dev   # 开发
uv build              # 构建 wheel（含 ONNX 模型）
\`\`\`

## 使用

\`\`\`
bardecode [OPTIONS] IMAGE [IMAGE ...]

选项：
  --conf-threshold FLOAT   检测置信度阈值（默认 0.25）
  --iou-threshold FLOAT    NMS IoU 阈值（默认 0.45）
  --img-size INT           推理输入尺寸（默认 640）
  --model PATH             ONNX 权重路径（默认用包内）
  --formats STR            逗号分隔码格式白名单（默认全部）
  --no-fallback            禁用全图回退解码
  -q, --quiet              仅打印码值
  -v, --verbose
  --version
\`\`\`

## 示例

\`\`\`powershell
bardecode receipt.jpg
bardecode --quiet *.png > all_codes.txt
bardecode --formats QR_CODE ticket.png
\`\`\`

## 架构

YOLOv8s-barcode ONNX（22MB，打包进 wheel）→ letterbox 预处理 → onnxruntime CPU 推理
→ NMS 后处理 → 裁剪 → zxing-cpp 解码 → JSON 输出。

模型来源：[Piero2411/YOLOV8s-Barcode-Detection](https://huggingface.co/Piero2411/YOLOV8s-Barcode-Detection)，
通过 \`scripts/export_onnx.py\` 导出。

## 测试

\`\`\`powershell
uv run pytest                 # 跳过 slow
uv run pytest -m slow         # 含模型推理的集成测试
\`\`\`
```

- [ ] **Step 2: 提交**

```powershell
git add README.md
git commit -m "docs: README with install/usage/architecture"
```

---

## Task 13: 最终验证

- [ ] **Step 1: 清洁环境重装**

```powershell
Remove-Item -Recurse -Force .venv
uv sync --extra dev
```

- [ ] **Step 2: 全量测试**

```powershell
uv run pytest -v
```

预期：所有测试 pass。

- [ ] **Step 3: 手动跑用户原图**

```powershell
uv run python -m bardecode tests\fixtures\top-barcodes-in-retail-header.jpg.webp
```

把输出 JSON 给用户确认码值是否正确。

- [ ] **Step 4: 构建 wheel 验证打包**

```powershell
uv build
```

检查 `dist/bardecode-0.1.0-*.whl` 包含 `bardecode/models/yolov8s_barcode.onnx`：

```powershell
uv run python -c "import zipfile; z=zipfile.ZipFile([str(p) for p in __import__('pathlib').Path('dist').glob('*.whl')][0]); print([n for n in z.namelist() if 'onnx' in n])"
```

预期：列表中包含 `bardecode/models/yolov8s_barcode.onnx`。

- [ ] **Step 5: 最终提交（如有改动）**

```powershell
git add -A
git commit -m "chore: final verification" --allow-empty
```
