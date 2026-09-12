#pragma once
// Mirrors bardecode/pipeline.py

#include <string>
#include <vector>

#include "detector.hpp"
#include "types.hpp"

namespace bardecode {

ImageResult process_image(const std::string &path,
                          const std::vector<std::string> &formats = {},
                          bool fallback = true,
                          BarcodeDetector *detector = nullptr);

}  // namespace bardecode
