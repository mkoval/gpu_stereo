#include <cmath>
#include <stdint.h>
#include <opencv2/gpu/devmem2d.hpp>

#define WSIZE 32

using cv::gpu::DevMem2D_;

namespace gpu {

template <typename Tsrc, typename Tker, typename Tdst>
__global__
void convolve(DevMem2D_<Tsrc> const src, DevMem2D_<Tker> const ker,
               DevMem2D_<Tdst> dst)
{
    int const r0 = blockDim.y * blockIdx.y + threadIdx.y;
    int const c0 = blockDim.x + blockIdx.x + threadIdx.x;
    int const r_offset = ker.rows / 2;
    int const c_offset = ker.cols / 2;
    Tdst response = 0;

    if (ker.rows / 2 <= r0 && r0 < src.rows - ker.rows / 2
     && ker.cols / 2 <= c0 && c0 < src.cols - ker.cols / 2) {
        for (int dr = 0; dr < ker.rows; dr++) {
            Tsrc const *const src_row = src.ptr(r0 + dr - r_offset);
            Tker const *const ker_row = ker.ptr(dr);

            for (int dc = 0; dc < ker.cols; dc++) {
                response += (Tdst)src_row[c0 + dc - c_offset] * ker_row[dc];
            }
        }
    }
    dst.ptr(r0)[c0] = response;
}

template <typename Tsrc, typename Tker, typename Tdst>
__host__
void convolve_caller(DevMem2D_<Tsrc> src, DevMem2D_<Tker> ker,
                     DevMem2D_<Tdst> dst)
{
    int const tpb = (int)floor(sqrt(WSIZE));
    dim3 Dg((src.cols + 1) / tpb, (src.rows + 1) / tpb);
    dim3 Db(tpb, tpb);
    convolve<Tsrc, Tker, Tdst><<<Dg, Db>>>(src, ker, dst);
}

template void convolve_caller<uint8_t, int8_t, int16_t>(
    DevMem2D_<uint8_t> src, DevMem2D_<int8_t> ker, DevMem2D_<int16_t> dst);

}
