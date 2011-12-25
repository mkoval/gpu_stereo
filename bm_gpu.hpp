#ifndef BM_GPU_HPP_
#define BM_GPU_HPP_

#include <stdint.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#include <opencv2/gpu/gpu.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace gpu {

void convolve(cv::gpu::GpuMat const &src, 
              cv::gpu::GpuMat const &ker,
              cv::gpu::GpuMat       &dst);

}

#endif
