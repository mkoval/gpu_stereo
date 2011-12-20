#ifndef UTIL_H_
#define UTIL_H_

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma clang diagnostic pop

void MatchBM(cv::Mat const &left, cv::Mat const &right, cv::Mat &disparity);

#endif
