#include <stdint.h>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "bm_gpu.hpp"

using cv::Mat;
using cv::Mat_;
using cv::gpu::DevMem2D_;
using cv::gpu::GpuMat;
using std::vector;

namespace gpu {

void convolve(GpuMat const &src, GpuMat const &ker, GpuMat &dst)
{
    CV_Assert(src.type() == CV_8UC1 && ker.type() == CV_8SC1);

    dst.create(src.rows, src.cols, CV_16SC1);
    convolve_caller<uint8_t, int8_t, int16_t>(src, ker, dst);
}

/***************************************************************************/

void sadbm(GpuMat const &left, GpuMat const &right, GpuMat &disparity)
{
    static Mat const kernel_log = (Mat_<int8_t>(9, 9) <<
        0, 1, 1,   2,   2,   2, 1, 1, 0,
        1, 2, 4,   5,   5,   5, 4, 2, 1,
        1, 4, 5,   3,   0,   3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5,   3,   0,   3, 5, 4, 1,
        1, 2, 4,   5,   5,   5, 4, 2, 1,
        0, 1, 1,   2,   2,   2, 1, 1, 0
    );
    static int const maxd = 64;
    static int const sad_rows = 21;
    static int const sad_cols = 21;
    GpuMat const kernel_log_gpu(kernel_log);

    CV_Assert(left.rows == right.rows);
    CV_Assert(left.type() == right.type());

    GpuMat left_log, right_log;
    gpu::convolve(left, kernel_log_gpu, left_log);
    gpu::convolve(right, kernel_log_gpu, right_log);

    // Compute SAD integral images for each disparity level.
    GpuMat integrals(left.rows * (maxd + 1), left.cols, CV_32SC1);
    for (int d = 0; d <= maxd; d++) {
        GpuMat integral = integrals.rowRange(d*left.rows, (d + 1)*left.rows);

        integral.create(left.rows, left.cols, CV_32SC1);
        sad_hor_caller<int16_t, int32_t>(left_log, right_log, integral, sad_cols, d);
        sad_ver_caller<int32_t>(integral);
    }

    disparity.create(left.rows, left.cols, CV_8UC1);
    disparity_picker_caller<int32_t, uint8_t>(integrals, disparity, left.rows,
                                              sad_rows, sad_cols, maxd);
}

}
