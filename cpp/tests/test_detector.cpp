#include <gtest/gtest.h>

#include "detector.hpp"

using namespace bardecode;

TEST(Letterbox, SquarePassthrough) {
    cv::Mat img(640, 640, CV_8UC3, cv::Scalar(1, 2, 3));
    auto lb = BarcodeDetector::letterbox(img, 640);
    EXPECT_EQ(lb.out.size(), cv::Size(640, 640));
    EXPECT_DOUBLE_EQ(lb.ratio, 1.0);
    EXPECT_EQ(lb.pad_w, 0);
    EXPECT_EQ(lb.pad_h, 0);
}

TEST(Letterbox, LandscapeGetsSidePadding) {
    cv::Mat img(200, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    auto lb = BarcodeDetector::letterbox(img, 640);
    EXPECT_EQ(lb.out.size(), cv::Size(640, 640));
    EXPECT_DOUBLE_EQ(lb.ratio, 1.0);     // width already 640, no scaling
    EXPECT_EQ(lb.pad_w, 0);
    EXPECT_EQ(lb.pad_h, 220);            // (640-200)/2
    // padding is gray 114
    EXPECT_EQ(lb.out.at<cv::Vec3b>(0, 0), cv::Vec3b(114, 114, 114));
}

TEST(Postprocess, FiltersLowConfidence) {
    // rows = 4+1 (nc=1), cols = 2; first box confident, second not
    std::vector<float> raw{
        // cx, cy rows (per-column layout: row-major [rows, cols])
        10, 50,
        10, 10,
        20, 20,
        20, 20,
        0.9, 0.1,
    };
    auto dets = BarcodeDetector::postprocess(raw, 5, 2, 0.25, 0.45);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].x1, 0);
    EXPECT_EQ(dets[0].y1, 0);
    EXPECT_EQ(dets[0].x2, 20);
    EXPECT_EQ(dets[0].y2, 20);
    EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
}

TEST(Postprocess, NmsSuppressesOverlap) {
    // rows = 4+1, cols = 2; two nearly identical confident boxes
    std::vector<float> raw{
        100, 101,
        100, 101,
        50,  50,
        50,  50,
        0.9, 0.8,
    };
    auto dets = BarcodeDetector::postprocess(raw, 5, 2, 0.25, 0.45);
    EXPECT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
}

TEST(Postprocess, EmptyWhenNoConfident) {
    std::vector<float> raw{
        10, 50,
        10, 10,
        20, 20,
        20, 20,
        0.9, 0.1,
    };
    auto dets = BarcodeDetector::postprocess(raw, 5, 2, 0.95, 0.45);
    EXPECT_TRUE(dets.empty());
}
