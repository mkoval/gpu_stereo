#include <cmath>
#include <cstdio>
#include <stdint.h>
#include <opencv2/gpu/devmem2d.hpp>

#define WSIZE 32

using cv::gpu::DevMem2D_;

template <typename T>
__host__ __device__
static T abs(T const &x)
{
    return (x >= 0) ? x : -x;
}

namespace gpu {

template <typename Tsrc, typename Tker, typename Tdst>
__global__
void convolve(DevMem2D_<Tsrc> const src, DevMem2D_<Tker> const ker,
               DevMem2D_<Tdst> dst)
{
    int const r0 = blockDim.y * blockIdx.y + threadIdx.y;
    int const c0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const r_offset = ker.rows / 2;
    int const c_offset = ker.cols / 2;

    if (r_offset <= r0 && r0 < src.rows - r_offset
     && c_offset <= c0 && c0 < src.cols - c_offset) {
        Tdst response = 0;

        for (int dr = 0; dr < ker.rows; dr++) {
            Tsrc const *const src_row = src.ptr(r0 + dr - r_offset);
            Tker const *const ker_row = ker.ptr(dr);

            for (int dc = 0; dc < ker.cols; dc++) {
                response += (Tdst)src_row[c0 + dc - c_offset] * ker_row[dc];
            }
        }

        dst.ptr(r0)[c0] = response;
    }
}

template <typename Tsrc, typename Tker, typename Tdst>
__host__
void convolve_caller(DevMem2D_<Tsrc> src, DevMem2D_<Tker> ker,
                     DevMem2D_<Tdst> dst)
{
    int const tpb = (int)sqrt(WSIZE);
    dim3 const Dg((src.cols + 1)/tpb, (src.rows + 1)/tpb);
    dim3 const Db(tpb, tpb);
    convolve<Tsrc, Tker, Tdst><<<Dg, Db>>>(src, ker, dst);
}

template __host__ void convolve_caller<uint8_t, int8_t, int16_t>(
    DevMem2D_<uint8_t> src, DevMem2D_<int8_t> ker, DevMem2D_<int16_t> dst);

/****************************************************************************/

template <typename Tsrc, typename Tdst>
__global__
void sad_hor(DevMem2D_<Tsrc> left, DevMem2D_<Tsrc> right, DevMem2D_<Tdst> sad,
             int window_cols, int disparity)
{
    int const r0 = blockDim.y * blockIdx.y + threadIdx.y;
    int const c0 = blockDim.x * blockIdx.x + threadIdx.x;
    Tsrc const *const left_row  = left.ptr(r0);
    Tsrc const *const right_row = right.ptr(r0);
    int const offset = window_cols / 2;

    if (offset + disparity <= c0 && c0 < left.cols - offset) {
        Tdst sum = 0;
        for (int dc = 0; dc < window_cols; dc++) {
            Tsrc const left_px  = left_row[c0 + dc - offset];
            Tsrc const right_px = right_row[c0 + dc - offset - disparity];
            sum += abs<Tdst>((Tdst)left_px - (Tdst)right_px);
        }
        sad.ptr(r0)[c0] = sum;
    } else if (0 <= c0 && c0 < left.cols) {
        sad.ptr(r0)[c0] = 0;
    }
}

template <typename Tsrc, typename Tdst>
__host__
void sad_hor_caller(DevMem2D_<Tsrc> left, DevMem2D_<Tsrc> right,
                     DevMem2D_<Tdst> sad, int window_cols, int disparity)
{
    int const tpb = (int)sqrt(WSIZE);
    dim3 const Dg((left.cols + 1)/tpb, (left.rows + 1)/tpb);
    dim3 const Db(tpb, tpb);
    sad_hor<Tsrc, Tdst><<<Dg, Db>>>(left, right, sad, window_cols, disparity);
}

template __host__ void sad_hor_caller<int16_t, int32_t>(
    DevMem2D_<int16_t> left, DevMem2D_<int16_t> right, DevMem2D_<int32_t> sad,
    int window_cols, int disparity);

}
