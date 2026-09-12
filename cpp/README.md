# bardecode-cpp

Python 版 `bardecode` 的 C++17 重写，用于学习 onnxruntime / OpenCV / zxing-cpp 原生
API 与 CMake 工程。功能与 Python 版对齐：YOLOv8 ONNX 检测条码 → 裁剪 → zxing-cpp
解码 → JSON 输出。模型直接复用 `../bardecode/models/yolov8s_barcode.onnx`。

## 模块对照

| C++ 文件 | Python 对应 | 说明 |
|---|---|---|
| `src/types.hpp` | `schemas.py` | Detection / DecodedBarcode / ImageResult / Report |
| `src/image_io.{hpp,cpp}` | `image_io.py` | cv::imread + nlohmann/json 报告（Pillow 兜底不需要，OpenCV codec 已覆盖） |
| `src/detector.{hpp,cpp}` | `detector.py` | letterbox（640×640 灰 114）、BGR→RGB、/255、NCHW、onnxruntime 推理、class-agnostic NMS、坐标反映射 |
| `src/decoder.{hpp,cpp}` | `decoder.py` | zxing::ReadBarcodes + 格式白名单（`BarcodeFormatsFromString`） |
| `src/pipeline.{hpp,cpp}` | `pipeline.py` | 裁剪解码、全图兜底、去重、IoU 清理 undecoded |
| `src/main.cpp` | `__main__.py` | CLI，退出码 0/1/2（2 = file not found） |

## 构建（Windows + VS2026）

依赖全部由 CMake FetchContent 拉取，无需 vcpkg：

- nlohmann/json v3.11.3（头文件）
- GoogleTest v1.15.2（仅测试）
- zxing-cpp v2.2.1（源码编译）
- OpenCV 4.10.0（源码编译，仅 core/imgproc/imgcodecs/dnn）
- onnxruntime 1.20.1（nuget 官方预编译包，免源码编译）

```bash
cmake -S cpp -B cpp/build -G "Visual Studio 18 2026" -A x64
cmake --build cpp/build --config Release   # 首次会编译 OpenCV，耗时较长
ctest --test-dir cpp/build -C Release --output-on-failure
```

可执行文件在 `cpp/build/Release/bardecode.exe`，模型由构建系统拷贝到其旁的
`models/` 目录。

### 网络说明

本仓库所在网络无法直连 `github.com`。CMakeLists 用 `codeload.github.com` 的
tarball 拉取源码依赖，onnxruntime 走 nuget.org。zxing-cpp 的示例目录还会尝试从
github 拉 stb，已通过 `FETCHCONTENT_SOURCE_DIR_STB` 预下载覆盖。若你所在网络可直连
GitHub，这些 override 不影响正常构建。

## 使用

```bash
./bardecode.exe image1.png image2.jpg            # JSON 报告
./bardecode.exe -q image.png                     # 只打印条码内容
./bardecode.exe --formats EAN13,QRCode image.png # 格式白名单
./bardecode.exe --model path/to/model.onnx --conf-threshold 0.4 image.png
echo $?    # 0 成功 / 1 处理错误 / 2 文件不存在
```

## 测试

```bash
uv run python cpp/tests/make_fixtures.py   # 离线生成测试图（EAN-13、QR、空白图）
ctest --test-dir cpp/build -C Release --output-on-failure
```
