#ifndef BM_CPU_H_
#define BM_CPU_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif

#include <opencv2/opencv.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace cpu {

void LaplacianOfGaussian(cv::Mat const &src, cv::Mat &dst);

void MatchBM(cv::Mat const &left, cv::Mat const &right, cv::Mat &disparity,
             int ndisparities, int window_size);

}

#endif
