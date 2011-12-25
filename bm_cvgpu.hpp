#ifndef BM_CVGPU_HPP_
#define BM_CVGPU_HPP_

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

namespace gpu {

void LaplacianOfGaussian(cv::Mat const &src, cv::Mat &dst);

void StereoBM(cv::Mat const &left, cv::Mat const &right, cv::Mat &disparity);

}

#endif
