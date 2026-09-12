#include "pipeline.hpp"

#include <set>
#include <tuple>

#include "decoder.hpp"
#include "image_io.hpp"

namespace bardecode {

static double iou(const std::array<int, 4> &a, const std::array<int, 4> &b) {
    int ix1 = std::max(a[0], b[0]), iy1 = std::max(a[1], b[1]);
    int ix2 = std::min(a[2], b[2]), iy2 = std::min(a[3], b[3]);
    int iw = std::max(0, ix2 - ix1), ih = std::max(0, iy2 - iy1);
    double inter = double(iw) * ih;
    if (inter == 0) return 0.0;
    double aarea = double(a[2] - a[0]) * (a[3] - a[1]);
    double barea = double(b[2] - b[0]) * (b[3] - b[1]);
    return inter / (aarea + barea - inter);
}

ImageResult process_image(const std::string &path,
                          const std::vector<std::string> &formats, bool fallback,
                          BarcodeDetector *detector) {
    ImageResult ir;
    ir.image = path;

    cv::Mat img;
    try {
        img = read_image(path);
    } catch (const std::exception &e) {
        ir.error = e.what();  // "file not found" from read_image
        return ir;
    }

    ir.width = img.cols;
    ir.height = img.rows;

    std::vector<Detection> detections;
    try {
        static BarcodeDetector shared_detector;  // lazy singleton, mirrors Python
        BarcodeDetector &det = detector ? *detector : shared_detector;
        detections = det.detect(img);
    } catch (const std::exception &e) {
        ir.error = std::string("detect error: ") + e.what();
        return ir;
    }

    for (const auto &d : detections) {
        cv::Mat crop = img(cv::Rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));
        if (crop.empty()) continue;
        std::vector<Decoded> results;
        try {
            results = decode_region(crop, formats);
        } catch (const std::exception &) {
        }
        if (!results.empty()) {
            for (const auto &r : results) {
                DecodedBarcode b;
                b.format = r.format;
                b.text = r.text;
                b.bbox = {d.x1, d.y1, d.x2, d.y2};
                b.detection_score = d.score;
                ir.barcodes.push_back(std::move(b));
            }
        } else {
            ir.undecoded.push_back({{d.x1, d.y1, d.x2, d.y2}, d.score});
        }
    }

    if (fallback && ir.barcodes.empty()) {
        std::vector<Decoded> full_results;
        try {
            full_results = decode_region(img, formats);
        } catch (const std::exception &) {
        }
        for (const auto &r : full_results) {
            int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
            for (const auto &[x, y] : r.position) {
                minx = std::min(minx, x); miny = std::min(miny, y);
                maxx = std::max(maxx, x); maxy = std::max(maxy, y);
            }
            std::array<int, 4> fb{minx, miny, maxx, maxy};
            DecodedBarcode b;
            b.format = r.format;
            b.text = r.text;
            b.bbox = fb;
            b.detection_score = 0.0;
            ir.barcodes.push_back(std::move(b));
            // Drop overlapping undecoded entries (we now have the decoded version)
            std::vector<UndecodedItem> kept;
            for (const auto &u : ir.undecoded) {
                if (iou(u.bbox, fb) <= 0.5) kept.push_back(u);
            }
            ir.undecoded = std::move(kept);
        }
    }

    // Dedup identical (format, text, bbox)
    std::set<std::tuple<std::string, std::string, std::array<int, 4>>> seen;
    std::vector<DecodedBarcode> deduped;
    for (auto &b : ir.barcodes) {
        auto key = std::make_tuple(b.format, b.text, b.bbox);
        if (seen.insert(key).second) deduped.push_back(std::move(b));
    }
    ir.barcodes = std::move(deduped);

    return ir;
}

}  // namespace bardecode
