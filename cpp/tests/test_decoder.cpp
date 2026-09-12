#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include "decoder.hpp"

using namespace bardecode;

TEST(DecodeRegion, ReadsEan13) {
    cv::Mat img = cv::imread(std::string(FIXTURES_DIR) + "/ean13.png");
    ASSERT_FALSE(img.empty());
    auto results = decode_region(img);
    ASSERT_EQ(results.size(), 1u);
    // C++ zxing-cpp reports "EAN-13" (Python bindings use "EAN13")
    EXPECT_EQ(results[0].format, "EAN-13");
    EXPECT_EQ(results[0].text, "4006381333931");
    EXPECT_EQ(results[0].position.size(), 4u);
}

TEST(DecodeRegion, FormatWhitelist) {
    cv::Mat img = cv::imread(std::string(FIXTURES_DIR) + "/ean13.png");
    ASSERT_FALSE(img.empty());
    // EAN13 whitelisted -> decodes
    auto ok = decode_region(img, {"EAN13"});
    EXPECT_EQ(ok.size(), 1u);
    // QRCode-only whitelist -> nothing
    auto none = decode_region(img, {"QRCode"});
    EXPECT_TRUE(none.empty());
}

TEST(DecodeRegion, InvalidFormatThrows) {
    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    EXPECT_THROW(decode_region(img, {"NoSuchFormat"}), std::invalid_argument);
}

TEST(DecodeRegion, BlankImageYieldsNothing) {
    cv::Mat img = cv::imread(std::string(FIXTURES_DIR) + "/blank.png");
    ASSERT_FALSE(img.empty());
    auto results = decode_region(img);
    EXPECT_TRUE(results.empty());
}
