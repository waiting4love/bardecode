# bardecode

命令行工具：从照片中检测并解码所有 1D/2D 条形码。CPU 推理，针对 x86 小主机优化。

## 工作原理

```
照片 → YOLOv8s-barcode ONNX 检测（onnxruntime CPU）
     → 每个检测区域裁剪
     → zxing-cpp 解码
     → JSON 输出
```

- **检测模型**：YOLOv8s 微调权重（条形码检测），导出为 ONNX（42.7 MB），打包进 wheel，运行时无需 PyTorch、无需联网。
- **解码器**：zxing-cpp，支持 EAN/UPC、Code128/39/93、Codabar、ITF、QR、DataMatrix、Aztec、PDF417 等。

## 安装

```powershell
# 开发
uv sync --extra dev

# 构建 wheel（含 ONNX 模型）
uv build
```

运行时依赖：`onnxruntime`、`zxing-cpp`、`opencv-python-headless`、`numpy`、`Pillow`。

## 使用

```
bardecode [OPTIONS] IMAGE [IMAGE ...]

参数：
  IMAGE                   一个或多个图片路径

选项：
  --conf-threshold FLOAT  检测置信度阈值（默认 0.25）
  --iou-threshold FLOAT   NMS IoU 阈值（默认 0.45）
  --img-size INT          推理输入尺寸（默认 640）
  --model PATH            ONNX 权重路径（默认用包内打包的模型）
  --formats STR           逗号分隔的码格式白名单（zxing-cpp 枚举名，如 EAN13,QRCode）
  --no-fallback           禁用全图回退解码
  -q, --quiet             仅打印码值，每行一个
  -v, --verbose           诊断信息打印到 stderr
  --version               显示版本号
```

退出码：`0` 全部成功，`1` 有一般错误，`2` 有文件未找到（取最严重的）。

## 示例

```powershell
# 默认 JSON 输出
bardecode receipt.jpg

# 多张图，只输出码值
bardecode --quiet *.png > all_codes.txt

# 只要 QR 码
bardecode --formats QRCode ticket.png

# 调低置信度阈值以找更多候选
bardecode --conf-threshold 0.1 blurry.jpg
```

## 输出格式（JSON）

```json
{
  "tool": "bardecode",
  "version": "0.1.0",
  "results": [
    {
      "image": "receipt.jpg",
      "width": 1526,
      "height": 720,
      "barcodes": [
        {
          "format": "EAN-13",
          "text": "0065515000037",
          "bbox": [2, 54, 350, 548],
          "detection_score": 0.857
        }
      ],
      "undecoded": [
        {"bbox": [743, 276, 1123, 636], "detection_score": 0.791}
      ],
      "error": null
    }
  ]
}
```

- `barcodes`：成功解码的条形码（含格式、文本、检测框、检测置信度）。
- `undecoded`：检测到但未能解码的区域（可能是非条形码纹理，或画质不足）。
- `error`：该图片处理出错时填（如 `"file not found"`），否则为 `null`。

## 模型来源与再导出

检测权重来自 [Piero2411/YOLOV8s-Barcode-Detection](https://huggingface.co/Piero2411/YOLOV8s-Barcode-Detection)，通过一次性脚本导出为 ONNX：

```powershell
uv run --with ultralytics --with requests python scripts/export_onnx.py
```

这会下载 `.pt`、导出 ONNX 到 `bardecode/models/yolov8s_barcode.onnx`、删除中间 `.pt`。仅开发期需要 ultralytics/PyTorch；运行时只用 onnxruntime。

## 测试

```powershell
uv run pytest                 # 全部测试（含 slow 集成测试，需加载模型）
uv run pytest -m "not slow"   # 仅快速单元测试（不加载模型）
```

## 许可证

本项目代码采用 **MIT** 协议（见 [LICENSE](LICENSE)）。

**模型权重声明**：内置的 `yolov8s_barcode.onnx` 来自 [Piero2411/YOLOV8s-Barcode-Detection](https://huggingface.co/Piero2411/YOLOV8s-Barcode-Detection)，基于 Ultralytics YOLOv8 架构（Ultralytics 以 AGPL-3.0 发布）。本仓库仅分发导出后的 ONNX 权重用于推理，不包含 Ultralytics 源代码。如对商用场景的协议合规有疑问，请自行评估或咨询法律意见。
