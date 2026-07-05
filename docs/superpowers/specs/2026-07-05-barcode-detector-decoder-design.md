# bardecode — 条形码检测与解码工具设计

- 日期：2026-07-05
- 状态：设计已确认，待实现

## 1. 目标

编写一个命令行工具 `bardecode`，接收一张或多张照片，从中找出**所有**条形码（含 1D 与 2D）并解码出文本。目标主机为 **CPU x86 小主机（8GB+ 内存）**，因此必须使用极轻量的 AI 检测模型与高效的解码后端。

## 2. 非目标（YAGNI）

- 不做批量目录扫描（用户用 shell glob 即可）。
- 不做 GUI、不做 Web 服务。
- 不做视频流处理（仅静态图片）。
- 不做 GPU 加速（CPU 已够用）。
- 不训练新模型（复用公开权重）。

## 3. 架构

### 3.1 数据流（单张图）

```
CLI 收图路径
  → 读图 (cv2.imread, 含 webp 兜底走 Pillow)
  → YOLOv8n ONNX 推理 (onnxruntime CPU)
  → 后处理：置信度过滤 + NMS + letterbox 反变换
  → 得到 N 个 bbox
  → 对每个 bbox 裁剪原图 → zxing-cpp 解码
  → 若裁剪图无结果且未禁用，回退全图调一次 zxing-cpp
  → 汇总 (format, text, bbox, score)
  → JSON 输出
```

### 3.2 模块拆分

每个文件单一职责，便于独立测试：

```
bardecode/
├── __init__.py
├── __main__.py          # CLI 入口 (argparse)
├── pipeline.py          # 编排：读图 → 检测 → 裁剪 → 解码 → 汇总
├── detector.py          # YOLOv8 ONNX 检测器封装 + 后处理
├── decoder.py           # zxing-cpp 解码器封装
├── model_cache.py       # 首次下载 + 缓存到用户缓存目录
├── image_io.py          # 输入枚举 + webp 兜底读取 + JSON 格式化
└── schemas.py           # dataclass: Detection / DecodedBarcode / ImageResult / Report
pyproject.toml           # uv 管理
tests/
├── test_detector.py
├── test_decoder.py
├── test_pipeline.py
├── test_image_io.py
├── test_cli.py
└── fixtures/
    ├── top-barcodes-in-retail-header.jpg.webp  # 用户提供的真实样本
    └── blank.png                                # 实现时生成的负样本
```

## 4. 依赖

**运行时**（最小集）：
- `onnxruntime`（CPU 版）
- `zxing-cpp`（≥ 2.0）
- `opencv-python-headless`（≥ 4.5；headless 避免拉 GUI 依赖）
- `numpy`
- `requests`（下载权重）
- `platformdirs`（跨平台缓存目录）

**开发依赖**：
- `pytest`
- `python-barcode`（生成合成测试码）
- `Pillow`（webp 兜底 + 生成测试负样本）

**虚拟环境**：使用 `uv` 创建与管理。

## 5. AI 检测器

### 5.1 模型

- 架构：YOLOv8n，输入 640×640，类别数 = 1（"barcode"）。
- 格式：ONNX（可直接被 onnxruntime 加载，无需 PyTorch）。
- 输出张量形状：`[1, 84, 8400]` 之类（实现时按实际权重核对，以代码为准）。

### 5.2 权重源策略（首次自动下载）

- 默认从 HuggingFace Hub 拉取一个开源 barcode-detector ONNX 权重。
- **候选权重源**（实现阶段需用 `webfetch` 验证可用性，最终锁定一个）：
  1. HuggingFace：`stephero/yolov8n-barcode` 一类（需验证存在性）
  2. HuggingFace：其他社区 barcode-detection YOLOv8n ONNX 仓库
  3. 兜底占位：通用 `yolov8n.pt`（COCO，无 barcode 类）— 仅用于开发期跑通管线，**正式版必须替换**
- 缓存路径：`platformdirs.user_cache_dir("bardecode") / "models/yolov8n_barcode.onnx"`。
- 失败回退：打印明确错误 + 提示用环境变量 `BARDECODE_MODEL_URL` 手动指定另一 URL。
- 提供 `bardecode download-model` 子命令做预下载（无网络环境下先备好）。
- 实现阶段第一件事是验证权重源；若候选都不可用，回到 brainstorming 重新讨论是否换用 OpenCV 内置 `BarcodeDetector` 或别的方案。

### 5.3 推理流程（detector.py 内部）

1. `cv2.imread` → BGR `numpy.ndarray`。
2. letterbox 到 640×640（保持比例，灰色 `(114,114,114)` 填充）。
3. BGR→RGB，归一化 `/255`，HWCN→NCHW，`astype(float32)`。
4. `session.run([], {input_name: ...})` 得到输出张量。
5. 过滤 `score < conf_threshold`（默认 0.25）。
6. NMS（IoU 阈值 0.45，用 `cv2.dnn.NMSBoxes`）。
7. 把 640 坐标 letterbox 反变换回原图坐标。
8. 返回 `List[Detection(x1, y1, x2, y2, score)]`。

## 6. 解码器

### 6.1 zxing-cpp 用法

```python
from zxingcpp import read_barcodes
results = read_barcodes(image_np)  # 直接吃 numpy（BGR 或灰度）
# 每个 result: .format / .text / .position (4 个点)
```

### 6.2 裁剪 → 解码策略

- 按检测 bbox 裁剪原图区域（BGR），送 `read_barcodes`。
- 单区域内若返回多个解码结果（罕见），全部保留。
- 每个解码结果同时附上：检测置信度（来自 YOLO）+ 解码位置（来自 zxing）。

### 6.3 双重兜底

- 若裁剪图无结果且未禁用（`--no-fallback`），回退用**全图**调一次 `read_barcodes`（zxing 自带检测会兜底漏检）。
- 若全图也解不出但检测器有输出，仍把 `bbox + score` 列入该图的 `undecoded` 列表，便于调试。
- 全图回退的解码结果需要去重（与裁剪解码结果 bbox IoU > 0.5 的视为同一个）。

### 6.4 支持的码格式

zxing-cpp 默认全开，可用 `--formats` 收窄：

- 1D：UPC-A/E、EAN-8/13、Code39/93/128、Codabar、ITF、RSS
- 2D：QR、DataMatrix、Aztec、PDF417

## 7. CLI 接口

```
bardecode [OPTIONS] IMAGE [IMAGE ...]

参数：
  IMAGE                   一个或多个图片路径（shell glob 由用户展开）

选项：
  --conf-threshold FLOAT  检测置信度阈值（默认 0.25）
  --iou-threshold FLOAT   NMS IoU 阈值（默认 0.45）
  --img-size INT          推理输入尺寸（默认 640）
  --model PATH            ONNX 权重路径（默认用缓存内）
  --formats STR           逗号分隔的码格式白名单（默认全部）
  --no-fallback           禁用全图回退解码
  -q, --quiet             仅打印码值，每行一个
  -v, --verbose           诊断信息打印到 stderr
  --version

子命令：
  download-model          预下载模型到缓存
```

## 8. 输出 JSON Schema（默认）

```json
{
  "tool": "bardecode",
  "version": "0.1.0",
  "results": [
    {
      "image": "top-barcodes-in-retail-header.jpg.webp",
      "width": 1200,
      "height": 600,
      "barcodes": [
        {
          "format": "EAN_13",
          "text": "4006381333918",
          "bbox": [x1, y1, x2, y2],
          "detection_score": 0.91
        }
      ],
      "undecoded": [
        {"bbox": [x1, y1, x2, y2], "detection_score": 0.42}
      ],
      "error": null
    }
  ]
}
```

`-q / --quiet` 模式只输出每个解码成功的码值，每行一个，不含 JSON 外壳。

## 9. 错误处理与退出码

- 图片读不到 → 该项 `error = "file not found"`，**继续处理其他图**，退出码 = 2。
- 模型缓存下载失败 → 退出码 = 3，打印明确指引（含 `BARDECODE_MODEL_URL` 提示）。
- 推理 / 解码异常 → 该项 `error = "<详情>"`，退出码 = 1。
- 全部成功 → 退出码 0。
- 顶层 try/except 包装 CLI，**禁止任何未捕获异常 crash 整个进程**。

多个错误同时存在时，退出码按"严重度最高"取：`3 > 2 > 1 > 0`。

## 10. 测试策略（pytest）

| 测试 | 类型 | 内容 |
|---|---|---|
| `letterbox` 几何 | 单元 | 保持比例、正确填充、可逆 |
| 后处理坐标变换 | 单元 | 640 坐标 → 原图坐标正确 |
| NMS 行为 | 单元 | 高 IoU 框被合并 |
| JSON schema | 单元 | 必填字段、类型 |
| zxing 合成码解码 | 单元 | 用 `python-barcode` 生成图后能解 |
| webp 读取 | 集成 | 用户提供的 webp 能被读入 |
| 端到端（webp 图）| 集成 | 至少检测到 1 个码 |
| 端到端（负样本）| 集成 | 空结果，不报错 |
| CLI 退出码 | subprocess | 0 / 1 / 2 / 3 各覆盖 |
| `download-model` | subprocess | 真实下载（标 slow，可跳过）|

**端到端测试**因依赖大模型，打 `slow` 标记，CI 默认跳过，本地 `pytest -m slow` 运行。单元测试**不依赖**模型权重。

## 11. 开发顺序建议

1. 用 `uv init` 创建项目骨架，加依赖。
2. 写 `schemas.py` + `image_io.py`（无模型依赖）。
3. 写 `decoder.py` + 单元测试（合成码）。
4. 验证权重源（webfetch + 手动试下载一个 ONNX）。
5. 写 `detector.py` + 单元测试（letterbox、后处理、NMS）。
6. 写 `pipeline.py` + 端到端测试（webp 图）。
7. 写 `__main__.py` + CLI 测试。
8. 写 `model_cache.py` + `download-model` 子命令。

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 找不到合用的公开 barcode ONNX 权重 | 兜底占位 + `BARDECODE_MODEL_URL` 兜底；最坏回到 brainstorming |
| `zxing-cpp` 在 Windows 编译困难 | 它在 PyPI 有预编译 wheel；若安装失败回退到 `pyzbar`（仅 1D+QR） |
| webp 在某些 OpenCV 构建下不支持 | Pillow 兜底读取 |
| 弱 CPU 推理慢 | YOLOv8n + 640 输入足够轻；未来可加 `--img-size 320` |
