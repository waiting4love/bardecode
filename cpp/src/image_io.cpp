#include "image_io.hpp"

#include <nlohmann/json.hpp>

namespace bardecode {

using json = nlohmann::json;

static json image_result_to_json(const ImageResult &ir) {
    json out;
    out["image"] = ir.image;
    out["width"] = ir.width;
    out["height"] = ir.height;
    out["barcodes"] = json::array();
    for (const auto &b : ir.barcodes) {
        out["barcodes"].push_back({
            {"format", b.format},
            {"text", b.text},
            {"bbox", b.bbox},
            {"detection_score", b.detection_score},
        });
    }
    out["undecoded"] = json::array();
    for (const auto &u : ir.undecoded) {
        out["undecoded"].push_back({
            {"bbox", u.bbox},
            {"detection_score", u.detection_score},
        });
    }
    out["error"] = ir.error ? json(*ir.error) : json(nullptr);
    return out;
}

std::string format_report_json(const Report &report, bool quiet) {
    if (quiet) {
        std::string lines;
        for (const auto &ir : report.results) {
            for (const auto &b : ir.barcodes) {
                if (!lines.empty()) lines += "\n";
                lines += b.text;
            }
        }
        return lines;
    }
    json j;
    j["tool"] = report.tool;
    j["version"] = report.version;
    j["results"] = json::array();
    for (const auto &ir : report.results) {
        j["results"].push_back(image_result_to_json(ir));
    }
    return j.dump(2);
}

}  // namespace bardecode
