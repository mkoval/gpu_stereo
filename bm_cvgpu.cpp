#include <stdint.h>
#include <cstdlib>
#include "bm_gpu.hpp"
#include "bm_cvgpu.hpp"

using cv::Mat;

namespace gpu {

void LaplacianOfGaussian(Mat const &src, Mat &dst)
{
    CV_Assert(src.type() == CV_8UC1);

    dst.create(src.rows, src.cols, CV_16SC1);
    LaplacianOfGaussian<uint8_t, int16_t>(
        src.ptr<uint8_t>(0), dst.ptr<int16_t>(0),
        src.step[0], dst.step[0],
        src.rows, src.cols
    );
}

void StereoBM(cv::Mat const &left, cv::Mat const &right, cv::Mat &dst)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.rows == right.rows && left.cols == right.cols);

    dst.create(left.rows, left.cols, CV_16SC1);

    StereoBM<uint8_t, int16_t, int32_t, int16_t>(
        left.ptr<uint8_t>(0), right.ptr<uint8_t>(0), dst.ptr<int16_t>(0),
        left.step[0], right.step[0], dst.step[0],
        left.rows, left.cols
    );
}

}
