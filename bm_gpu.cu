#include <cmath>
#include <limits>
#include <stdint.h>
#include <cuda_runtime.h>
#include "bm_gpu.hpp"

using std::numeric_limits;

namespace gpu {

__device__ __constant__ int8_t const KERNEL_LOG[9][9] = {
    { 0, 1, 1,   2,   2,   2, 1, 1, 0 },
    { 1, 2, 4,   5,   5,   5, 4, 2, 1 },
    { 1, 4, 5,   3,   0,   3, 5, 4, 1 },
    { 2, 5, 3, -12, -24, -12, 3, 5, 2 },
    { 2, 5, 0, -24, -40, -24, 0, 5, 2 },
    { 2, 5, 3, -12, -24, -12, 3, 5, 2 },
    { 1, 4, 5,   3,   0,   3, 5, 4, 1 },
    { 1, 2, 4,   5,   5,   5, 4, 2, 1 },
    { 0, 1, 1,   2,   2,   2, 1, 1, 0 }
};

#define KERNEL_LOG_ROWS 9
#define KERNEL_LOG_COLS 9
#define WSIZE 32

template <typename T>
__host__ __device__
static T &index(T *img, int r, int c, int pitch) {
    return *((T *)(((uint8_t *)img) + r * pitch + c));
}

template <typename T>
__host__ __device__
static T abs(T x) {
    return (x >= 0) ? x : -x;
}

template <typename Tin, typename Tker, typename Tout,
          size_t Krows, size_t Kcols>
__global__
void convolve(Tin const *const src, Tout *const dst, Tker const *const ker,
              int rows, int cols,
              size_t src_pitch, size_t dst_pitch)
{
    size_t const r0 = threadIdx.x + blockIdx.x * blockDim.x;
    size_t const c0 = threadIdx.y + blockIdx.y * blockDim.y;

    // This divergent execution path won't harm efficiency too much because one
    // of the paths terminates after a few clock cycles.
    if (Krows/2 <= r0 && r0 < rows - Krows/2
     && Kcols/2 <= c0 && c0 < cols - Kcols/2) {
        Tout value = 0;

        for (size_t dr = 0; dr < Krows; dr++)
        for (size_t dc = 0; dc < Kcols; dc++) {
            size_t const r = r0 + dr - Krows/2;
            size_t const c = c0 + dc - Kcols/2;
            value += ker[dr * Kcols + dc] * (Tout)index(src, r, c, src_pitch);
        }

        index(dst, r0, c0, dst_pitch) = value;
    } else {
        index(dst, r0, c0, dst_pitch) = 0;
    }
}

template <typename Tin, typename Tout>
__host__
void LaplacianOfGaussianD(Tin const *const srcd, Tout *const dstd,
                          size_t srcd_pitch, size_t dstd_pitch,
                          int rows, int cols)
{
    // Split the image into as many sqrt(WSIZE)*sqrt(WSIZE) blocks as is needed
    // to cover the entire image. This guarantees that there are approximately
    // WSIZE threads per block. Then, solve for the convolution using one thread
    // per pixel.
    int const tpb = (int)floor(sqrt(WSIZE));
    convolve<Tin, int8_t, Tout, KERNEL_LOG_ROWS, KERNEL_LOG_COLS>
            <<<dim3((cols + 1)/tpb, (rows + 1)/tpb), dim3(tpb, tpb)>>>
            (srcd, dstd, (int8_t const*)KERNEL_LOG, rows, cols, srcd_pitch, dstd_pitch);
}

template <typename Tin, typename Tout>
__host__
void LaplacianOfGaussian(Tin const *const src, Tout *const dst,
                         size_t src_pitch, size_t dst_pitch,
                         int rows, int cols)
{
    // Copy the input image to the device.
    Tin *srcd;
    size_t srcd_pitch;
    cudaMallocPitch(&srcd, &srcd_pitch, cols * sizeof(Tin), rows);
    cudaMemcpy2D(srcd, srcd_pitch, src, src_pitch,
                 cols * sizeof(Tin), rows, cudaMemcpyHostToDevice);

    // Allocate a buffer for the output on the device.
    Tout *dstd;
    size_t dstd_pitch;
    cudaMallocPitch(&dstd, &dstd_pitch, cols * sizeof(Tout), rows);

    LaplacianOfGaussianD<Tin, Tout>(
        srcd, dstd,
        srcd_pitch, dstd_pitch,
        rows, cols
    );

    // Copy the result back to the host.
    cudaMemcpy2D(dst, dst_pitch, dstd, dstd_pitch,
                 cols * sizeof(Tout), rows, cudaMemcpyDeviceToHost);
    cudaFree(srcd);
    cudaFree(dstd);
}

template void LaplacianOfGaussian<uint8_t, int16_t>(
    uint8_t const *const src, int16_t *const dst,
    size_t src_pitch, size_t dst_pitch,
    int rows, int cols);

/***************************************************************************/

#define DISPARITY_NONE 0
#define MAX_DISPARITY 64
#define SAD_ROWS 21
#define SAD_COLS 21

template <typename Tsrc, typename Tdst>
__global__
void HorizontalSAD(Tsrc const *const left, Tsrc const *const right, Tdst *const dst,
                   size_t left_pitch, size_t right_pitch, size_t dst_pitch,
                   int rows, int cols, int window_width, int disparity)
{
    int const r0 = threadIdx.x + blockIdx.x * blockDim.x;
    int const c0 = threadIdx.y + blockIdx.y * blockDim.y;
    Tdst &error = index(dst, r0, c0, dst_pitch);

    if (window_width/2 + disparity <= c0 && c0 < cols - window_width/2) {
        Tdst diff = 0;
        for (int dc = -window_width / 2; dc <= window_width / 2; dc++) {
            Tdst const left_px  = (Tdst)index<Tsrc const>(left,  r0, c0 + dc,             left_pitch);
            Tdst const right_px = (Tdst)index<Tsrc const>(right, r0, c0 + dc - disparity, right_pitch);
            diff += abs<Tdst>(left_px - right_px);
        }
        error = diff;
    } else {
        error = DISPARITY_NONE;
    }
}

template <typename T>
__global__
void VerticalIntegral(T *img, size_t img_pitch, int rows, int cols)
{
    int const c = threadIdx.x + blockIdx.x * blockDim.x;

    for (int r = 1; r < rows; r++) {
        T const &dst1_px = index(img, r - 1, c, img_pitch);
        T       &dst2_px = index(img, r - 0, c, img_pitch);
        dst2_px += dst1_px;
    }
}

template <typename Terr, typename Tdisp>
__global__
void BestDisparity(
    Terr *sadd, Tdisp *dst, size_t sadd_pitch, size_t dst_pitch,
    int rows, int cols, int sad_rows, int max_disparity, Terr max_error)
{
    int const r0 = threadIdx.x + blockIdx.x * blockDim.x;
    int const c0 = threadIdx.y + blockIdx.y * blockDim.y;


    Tdisp best_disp = 0;
    Terr  best_sad  = max_error;

    if (sad_rows/2 <= r0 && r0 < rows - sad_rows/2) {
        for (int d = 0; d <= max_disparity; d++) {
            Terr *sadd_lvl  = &index(sadd, d * rows, 0, sadd_pitch);
            Terr const sad1 = index(sadd_lvl, r0 - sad_rows/2, c0, sadd_pitch);
            Terr const sad2 = index(sadd_lvl, r0 + sad_rows/2, c0, sadd_pitch);
            Terr const sad  = sad2 - sad1;

            if (sad < best_sad) {
                best_disp = d;
                best_sad  = sad;
            }
        }
        index(dst, r0, c0, dst_pitch) = best_disp;
    } else {
        index(dst, r0, c0, dst_pitch) = DISPARITY_NONE;
    }
}

template <typename Tin, typename Tlog, typename Terr, typename Tdisp>
__host__
void StereoBM(Tin const *const left, Tin const *const right, Tdisp *const dst,
              size_t left_pitch, size_t right_pitch, size_t dst_pitch,
              int rows, int cols)
{
    int const tpb = (int)floor(sqrt(WSIZE));
    dim3 const Dg((cols + 1)/tpb, (rows + 1)/tpb);
    dim3 const Db(tpb, tpb);

    Tin *leftd, *rightd;
    size_t leftd_pitch, rightd_pitch;
    cudaMallocPitch(&leftd,  &leftd_pitch,  cols * sizeof(Tin), rows);
    cudaMallocPitch(&rightd, &rightd_pitch, cols * sizeof(Tin), rows);
    cudaMemcpy2D(leftd,  leftd_pitch,  left,  left_pitch,
                 cols * sizeof(Tin), rows, cudaMemcpyHostToDevice);
    cudaMemcpy2D(rightd, rightd_pitch, right, right_pitch,
                 cols * sizeof(Tin), rows, cudaMemcpyHostToDevice);

    // Pre-process both of the input images using the LoG.
    Tlog *leftd_log, *rightd_log;
    size_t leftd_log_pitch, rightd_log_pitch;
    cudaMallocPitch(&leftd_log,  &leftd_log_pitch,  cols * sizeof(Tlog), rows);
    cudaMallocPitch(&rightd_log, &rightd_log_pitch, cols * sizeof(Tlog), rows);
    LaplacianOfGaussianD<Tin, Tlog>(leftd,  leftd_log,  leftd_pitch,
                                    leftd_log_pitch,  rows, cols);
    LaplacianOfGaussianD<Tin, Tlog>(rightd, rightd_log, rightd_pitch,
                                    rightd_log_pitch, rows, cols);
    cudaFree(leftd);
    cudaFree(rightd);

    // Calculate the horizontal integral image for every possible disparity.
    Terr *sadd;
    size_t sadd_pitch;
    cudaMallocPitch(&sadd, &sadd_pitch, cols * sizeof(Terr),
                    rows * (MAX_DISPARITY + 1));
    for (int d = 0; d <= MAX_DISPARITY; d++) {
        Terr *sadd_level = &index(sadd, d * rows, 0, sadd_pitch);

        HorizontalSAD<Tlog, Terr><<<Dg, Db>>>(
            leftd_log, rightd_log, sadd_level,
            left_pitch, right_pitch, sadd_pitch,
            rows, cols, SAD_COLS, 0
        );
        VerticalIntegral<Terr><<<(cols + 1)/tpb, tpb>>>(
            sadd_level, sadd_pitch, rows, cols
        );
    }
    cudaFree(leftd_log);
    cudaFree(rightd_log);

    //
    Tdisp *disparityd;
    size_t disparityd_pitch;
    cudaMallocPitch(&disparityd,  &disparityd_pitch,  cols * sizeof(Tdisp), rows);
    BestDisparity<Terr, Tdisp><<<Dg, Db>>>(
        sadd, disparityd,
        sadd_pitch, disparityd_pitch,
        rows, cols, SAD_ROWS, MAX_DISPARITY, 
        numeric_limits<Terr>::max());

    cudaMemcpy2D(dst, dst_pitch, disparityd, disparityd_pitch,
                 cols * sizeof(Tdisp), rows, cudaMemcpyDeviceToHost);

    // TODO: debug
    Terr *debug = new Terr[rows * cols];
    cudaMemcpy2D(debug, cols * sizeof(Terr), sadd, sadd_pitch,
                 cols * sizeof(Terr), rows, cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) {
        index(dst, r, c, dst_pitch) = (Tdisp)index(debug, r, c, cols * sizeof(Terr));
    }
    // TODO: debug

    // Clean up device memory.
    cudaFree(disparityd);
    cudaFree(sadd);
}

template void StereoBM<uint8_t, int16_t, int32_t, int16_t>(
    uint8_t const *const left, uint8_t const *const right, int16_t *const dst,
    size_t left_pitch, size_t right_pitch, size_t dst_pitch,
    int rows, int cols);

}
