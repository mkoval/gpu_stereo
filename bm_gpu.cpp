#include <stdint.h>
#include "bm_gpu.hpp"

using cv::gpu::DevMem2D_;
using cv::gpu::GpuMat;


namespace gpu {

template <typename Tsrc, typename Tker, typename Tdst>
void convolve_caller(DevMem2D_<Tsrc> src, DevMem2D_<Tker> ker,
                     DevMem2D_<Tdst> dst);

void convolve(GpuMat const &src, GpuMat const &ker, GpuMat &dst)
{

    CV_Assert(src.type() == CV_8UC1 && ker.type() == CV_8SC1);

    dst.create(src.rows, src.cols, CV_16SC1);
    convolve_caller<uint8_t, int8_t, int16_t>(src, ker, dst);
}

}
