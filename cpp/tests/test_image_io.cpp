#include <gtest/gtest.h>

#include "image_io.hpp"

using namespace bardecode;

TEST(ReadImage, MissingFileThrows) {
    EXPECT_THROW(read_image("no_such_file_xyz.png"), std::runtime_error);
    try {
        read_image("no_such_file_xyz.png");
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "file not found");
    }
}

TEST(FormatReport, QuietPrintsValuesOnly) {
    Report r{"bardecode", "0.1.0", {}};
    ImageResult ir;
    ir.image = "a.png";
    ir.width = 10;
    ir.height = 20;
    DecodedBarcode b;
    b.format = "EAN13";
    b.text = "4006381333931";
    ir.barcodes.push_back(b);
    r.results.push_back(ir);
    EXPECT_EQ(format_report_json(r, true), "4006381333931");
}

TEST(FormatReport, JsonContainsFields) {
    Report r{"bardecode", "0.1.0", {}};
    ImageResult ir;
    ir.image = "a.png";
    ir.error = "file not found";
    r.results.push_back(ir);
    std::string js = format_report_json(r, false);
    EXPECT_NE(js.find("\"tool\": \"bardecode\""), std::string::npos);
    EXPECT_NE(js.find("\"error\": \"file not found\""), std::string::npos);
    EXPECT_NE(js.find("\"width\": 0"), std::string::npos);
}
