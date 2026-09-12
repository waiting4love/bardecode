#pragma once
// Mirrors bardecode/detector.py — YOLOv8 ONNX detection via onnxruntime

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "types.hpp"

namespace bardecode {

// Default model path: <exe_dir>/models/yolov8s_barcode.onnx
// (CMake copies it next to the executable), overridable via BARDECODE_MODEL.
std::string default_model_path();

class BarcodeDetector {
public:
    BarcodeDetector(const std::string &model_path = "", int img_size = 640,
                    double conf_thres = 0.25, double iou_thres = 0.45);

    // img: BGR. Returns detections in ORIGINAL image coordinates.
    std::vector<Detection> detect(const cv::Mat &img);

    // Kept public for tests (mirrors module-level functions in Python).
    struct Letterboxed {
        cv::Mat out;
        double ratio = 1.0;   // src_size / dst_size
        int pad_w = 0;        // left padding in output space
        int pad_h = 0;        // top padding
    };
    static Letterboxed letterbox(const cv::Mat &img, int target = 640);
    static std::vector<Detection> postprocess(const std::vector<float> &raw, int rows,
                                              int cols, double conf_thres = 0.25,
                                              double iou_thres = 0.45);

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    int img_size_;
    double conf_thres_;
    double iou_thres_;
};

}  // namespace bardecode
