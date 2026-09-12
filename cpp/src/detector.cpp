#include "detector.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // keep windows.h from defining min/max macros
#include <windows.h>
#endif

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace bardecode {

std::string default_model_path() {
    if (const char *env = std::getenv("BARDECODE_MODEL")) {
        if (*env) return env;
    }
    char exe[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, exe, MAX_PATH);
    std::string exe_dir(exe, n ? n : 0);
    auto slash = exe_dir.find_last_of("\\/");
    exe_dir = (slash == std::string::npos) ? "." : exe_dir.substr(0, slash);
    return exe_dir + "\\models\\yolov8s_barcode.onnx";
}

BarcodeDetector::Letterboxed BarcodeDetector::letterbox(const cv::Mat &img, int target) {
    Letterboxed lb;
    int h = img.rows, w = img.cols;
    double scale = double(target) / std::max(h, w);
    int new_w = std::max(int(std::round(w * scale)), 1);
    int new_h = std::max(int(std::round(h * scale)), 1);

    cv::Mat resized;
    cv::resize(img, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);

    lb.pad_w = (target - new_w) / 2;
    lb.pad_h = (target - new_h) / 2;

    lb.out = cv::Mat(target, target, img.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(lb.out(cv::Rect(lb.pad_w, lb.pad_h, new_w, new_h)));

    lb.ratio = 1.0 / scale;  // src/dst
    return lb;
}

std::vector<Detection> BarcodeDetector::postprocess(const std::vector<float> &raw, int rows,
                                                    int cols, double conf_thres,
                                                    double iou_thres) {
    // raw layout: [rows, cols] row-major; rows = 4 + nc, cols = N.
    // boxes in row 0..3 (cxcywh), scores = max over class rows.
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::array<float, 4>> xyxy;
    boxes.reserve(cols);
    scores.reserve(cols);
    xyxy.reserve(cols);

    for (int c = 0; c < cols; ++c) {
        float score = -1.f;
        for (int r = 4; r < rows; ++r) {
            score = std::max(score, raw[r * cols + c]);
        }
        if (score < conf_thres) continue;
        float cx = raw[0 * cols + c], cy = raw[1 * cols + c];
        float bw = raw[2 * cols + c], bh = raw[3 * cols + c];
        float x1 = cx - bw / 2, y1 = cy - bh / 2;
        xyxy.push_back({x1, y1, x1 + bw, y1 + bh});
        // NMSBoxes wants [x, y, w, h] with (x, y) = top-left corner (see Python note).
        boxes.emplace_back(cvRound(x1), cvRound(y1), cvRound(bw), cvRound(bh));
        scores.push_back(score);
    }
    if (scores.empty()) return {};

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(conf_thres),
                      static_cast<float>(iou_thres), idx);

    std::vector<Detection> out;
    out.reserve(idx.size());
    for (int i : idx) {
        out.push_back(Detection{static_cast<int>(xyxy[i][0]), static_cast<int>(xyxy[i][1]),
                                static_cast<int>(xyxy[i][2]), static_cast<int>(xyxy[i][3]),
                                static_cast<double>(scores[i])});
    }
    return out;
}

BarcodeDetector::BarcodeDetector(const std::string &model_path, int img_size,
                                 double conf_thres, double iou_thres)
    : env_(ORT_LOGGING_LEVEL_WARNING, "bardecode"),
      img_size_(img_size), conf_thres_(conf_thres), iou_thres_(iou_thres) {
    std::string path = model_path.empty() ? default_model_path() : model_path;
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // onnxruntime's Windows API takes a wide-char path
    std::wstring wpath = std::filesystem::path(path).wstring();
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), options);
    input_name_ = session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions{})
                      .get();  // NOLINT: string copied before the holder goes out of scope
    // C++ API requires explicit output names (Python's run(None) fetches all)
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        output_names_.push_back(
            session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{}).get());
    }
}

std::vector<Detection> BarcodeDetector::detect(const cv::Mat &img) {
    auto lb = letterbox(img, img_size_);
    // BGR->RGB, /255, HWC->NCHW
    cv::Mat rgb, blob;
    cv::cvtColor(lb.out, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(blob, CV_32F, 1.0 / 255.0);

    std::vector<int64_t> shape{1, 3, img_size_, img_size_};
    std::vector<float> nchw(static_cast<size_t>(3) * img_size_ * img_size_);
    for (int y = 0; y < img_size_; ++y) {
        const float *row = blob.ptr<float>(y);  // interleaved RGB, 3 floats per pixel
        for (int x = 0; x < img_size_; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                nchw[((static_cast<size_t>(ch) * img_size_ + y) * img_size_) + x] =
                    row[x * 3 + ch];
            }
        }
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, nchw.data(), nchw.size(), shape.data(), shape.size());

    const char *input_names[] = {input_name_.c_str()};
    std::vector<const char *> out_names;
    for (const auto &n : output_names_) out_names.push_back(n.c_str());
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                 out_names.data(), out_names.size());

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto tshape = info.GetShape();          // e.g. [1, 4+nc, N]
    int rows = static_cast<int>(tshape[1]);
    int cols = static_cast<int>(tshape[2]);
    const float *data = outputs[0].GetTensorData<float>();
    std::vector<float> raw(data, data + static_cast<size_t>(rows) * cols);

    auto dets = postprocess(raw, rows, cols, conf_thres_, iou_thres_);

    int w = img.cols, h = img.rows;
    std::vector<Detection> out;
    out.reserve(dets.size());
    for (auto &d : dets) {
        int x1 = std::clamp(int((d.x1 - lb.pad_w) * lb.ratio), 0, w - 1);
        int y1 = std::clamp(int((d.y1 - lb.pad_h) * lb.ratio), 0, h - 1);
        int x2 = std::clamp(int((d.x2 - lb.pad_w) * lb.ratio), 0, w - 1);
        int y2 = std::clamp(int((d.y2 - lb.pad_h) * lb.ratio), 0, h - 1);
        if (x2 <= x1 || y2 <= y1) continue;
        out.push_back(Detection{x1, y1, x2, y2, d.score});
    }
    return out;
}

}  // namespace bardecode
