#include "decoder.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <BarcodeFormat.h>
#include <ReadBarcode.h>

namespace bardecode {

std::vector<Decoded> decode_region(const cv::Mat &image,
                                   const std::vector<std::string> &formats) {
    ZXing::ReaderOptions options;
    if (!formats.empty()) {
        std::string joined;
        for (const auto &f : formats) {
            if (!joined.empty()) joined += "|";
            joined += f;
        }
        try {
            options.setFormats(ZXing::BarcodeFormatsFromString(joined));
        } catch (const std::invalid_argument &) {
            throw std::invalid_argument("Unknown barcode format: '" + joined + "'");
        }
    }

    // zxing works on luminance; convert BGR/gray to 8-bit gray first.
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    cv::Mat cont;
    if (!gray.isContinuous()) {
        gray = gray.clone();
    }

    ZXing::ImageView view(gray.data, gray.cols, gray.rows,
                          ZXing::ImageFormat::Lum, static_cast<int>(gray.step));
    auto results = ZXing::ReadBarcodes(view, options);

    std::vector<Decoded> out;
    out.reserve(results.size());
    for (const auto &r : results) {
        Decoded d;
        d.format = ZXing::ToString(r.format());
        d.text = r.text();
        for (auto p : {r.position().topLeft(), r.position().topRight(),
                       r.position().bottomRight(), r.position().bottomLeft()}) {
            d.position.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        }
        out.push_back(std::move(d));
    }
    return out;
}

}  // namespace bardecode
