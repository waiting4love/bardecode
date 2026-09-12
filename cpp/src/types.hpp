#pragma once
// Data structures mirroring bardecode/schemas.py

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace bardecode {

struct Detection {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    double score = 0.0;
};

struct DecodedBarcode {
    std::string format;
    std::string text;
    std::array<int, 4> bbox{};   // x1, y1, x2, y2
    double detection_score = 0.0;
};

struct UndecodedItem {
    std::array<int, 4> bbox{};
    double detection_score = 0.0;
};

struct ImageResult {
    std::string image;
    int width = 0;
    int height = 0;
    std::vector<DecodedBarcode> barcodes;
    std::vector<UndecodedItem> undecoded;
    std::optional<std::string> error;
};

struct Report {
    std::string tool;
    std::string version;
    std::vector<ImageResult> results;
};

}  // namespace bardecode
