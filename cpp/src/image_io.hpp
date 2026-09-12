#pragma once
// Mirrors bardecode/image_io.py (the Pillow fallback is covered by OpenCV codecs)

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "types.hpp"

namespace bardecode {

// cv::imread first; throws FileNotFoundError-style exception if the file
// cannot be read. Returns a 3-channel BGR image.
inline cv::Mat read_image(const std::string &path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("file not found");  // matches Python error string
    }
    return img;
}

// Format a Report as JSON (indent 2). If quiet, only barcode texts, one per line.
std::string format_report_json(const Report &report, bool quiet = false);

}  // namespace bardecode
