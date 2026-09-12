#pragma once
// Mirrors bardecode/decoder.py — zxing-cpp native API wrapper

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace bardecode {

struct Decoded {
    std::string format;
    std::string text;
    std::vector<std::pair<int, int>> position;  // TL, TR, BR, BL
};

// Decode all barcodes visible in a BGR or grayscale image.
// formats: optional whitelist of zxing-cpp BarcodeFormat enum names
// ("EAN13", "QRCode", ...). Empty vector = accept all formats.
// Throws std::invalid_argument on an unknown format name.
std::vector<Decoded> decode_region(const cv::Mat &image,
                                   const std::vector<std::string> &formats = {});

}  // namespace bardecode
