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

void sadbm(cv::gpu::GpuMat const &left,
           cv::gpu::GpuMat const &right,
           cv::gpu::GpuMat       &disparity);

/*************************************************************************/

template <typename Tsrc, typename Tker, typename Tdst>
void convolve_caller(cv::gpu::DevMem2D_<Tsrc> src, cv::gpu::DevMem2D_<Tker> ker,
                     cv::gpu::DevMem2D_<Tdst> dst);

template <typename Tsrc, typename Tdst>
void sad_hor_caller(cv::gpu::DevMem2D_<Tsrc> left, cv::gpu::DevMem2D_<Tsrc> right,
                     cv::gpu::DevMem2D_<Tdst> sad,
                     int window_cols, int disparity);

template <typename T>
void sad_ver_caller(cv::gpu::DevMem2D_<T> sad);

}

#endif
