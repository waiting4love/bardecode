#include <gtest/gtest.h>

#include "detector.hpp"
#include "pipeline.hpp"

using namespace bardecode;

TEST(ProcessImage, FileNotFound) {
    auto ir = process_image("no_such_file_xyz.png");
    ASSERT_TRUE(ir.error.has_value());
    EXPECT_EQ(*ir.error, "file not found");
    EXPECT_EQ(ir.width, 0);
    EXPECT_TRUE(ir.barcodes.empty());
}

TEST(ProcessImage, EndToEndBarcodePhoto) {
    auto ir = process_image(std::string(FIXTURES_DIR) + "/ean13_photo.png", {}, true);
    ASSERT_FALSE(ir.error.has_value());
    EXPECT_EQ(ir.width, 800);
    EXPECT_EQ(ir.height, 600);
    ASSERT_EQ(ir.barcodes.size(), 1u);
    EXPECT_EQ(ir.barcodes[0].format, "EAN-13");  // C++ zxing format name
    EXPECT_EQ(ir.barcodes[0].text, "4006381333931");
}

TEST(ProcessImage, FallbackDecodesFullImage) {
    // blank.png has no detectable barcode; fallback path still yields nothing
    auto ir = process_image(std::string(FIXTURES_DIR) + "/blank.png", {}, true);
    ASSERT_FALSE(ir.error.has_value());
    EXPECT_TRUE(ir.barcodes.empty());
}
