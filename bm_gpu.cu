#include <cmath>
#include <vector>
#include <stdint.h>
#include <cuda_runtime.h>
#include "bm_gpu.hpp"

using std::vector;

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

template <typename T> __host__ __device__ T abs(T x)
{
    return (x >= 0) ? x : -x;
}

template <typename Tin, typename Tker, typename Tout,
          size_t Krows, size_t Kcols>
__global__
void convolve(Tin const *const src, Tout *const dst, Tker const *const ker,
              int rows, int cols, size_t src_pitch, size_t dst_pitch)
{
    int const r0 = threadIdx.x + blockIdx.x * blockDim.x;
    int const c0 = threadIdx.y + blockIdx.y * blockDim.y;

    // This divergent execution path won't harm efficiency too much because one
    // of the paths terminates after a few clock cycles.
    if (Krows/2 <= r0 && r0 < rows - Krows/2
     && Kcols/2 <= c0 && c0 < cols - Kcols/2) {
        Tout value = 0;

        for (int dr = 0; dr < Krows; dr++)
        for (int dc = 0; dc < Kcols; dc++) {
            int const r = r0 + dr - Krows/2;
            int const c = c0 + dc - Kcols/2;
            value += ker[dr * Kcols + dc] * (Tout)src[r * src_pitch + c];
        }

        dst[r0 * dst_pitch + c0] = value;
    }
    // Some blocks may extend off the edge of the image if either of its
    // dimensions are not evenly divisible by WSIZE. This additional check
    // prevents writing to invalid memory.
    else if (0 <= r0 && r0 < rows && 0 <= c0 && c0 < cols) {
        dst[r0 * dst_pitch + c0] = 0;
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

    cudaMemcpy2D(dst, dst_pitch, dstd, dstd_pitch,
                 cols * sizeof(Tout), rows, cudaMemcpyDeviceToHost);

    cudaFree(srcd);
    cudaFree(dstd);
}

template void LaplacianOfGaussian(uint8_t const *const src, int16_t *const dst,
                                  size_t src_pitch, size_t dst_pitch,
                                  int rows, int cols);

template <typename Tin, typename Terr>
__global__
void StereoHorSAD(Tin const *const left, Tin const *const right, Terr *const horsad,
                  size_t left_pitch, size_t right_pitch, size_t horsad_pitch,
                  int window_cols, int rows, int cols, int d)
{
    int const r  = threadIdx.y + blockIdx.y * blockDim.y;
    int const c0 = threadIdx.x + blockIdx.x * blockDim.x;
    Terr error = 0;

    if (window_cols/2 + d <= c0 && c0 < cols - window_cols/2) {
        for (int dc = 0; dc < window_cols; dc++) {
            int const c_left  = c0 + dc - window_cols/2;
            int const c_right = c0 + dc - d - window_cols/2;
            error += abs<Terr>((Terr)left[r * left_pitch + c_left]
                             - (Terr)right[r * right_pitch + c_right]);
        }
    }
    horsad[r * horsad_pitch + c0] = error;
}

template <typename Tin, typename Tlog, typename Terr, typename Tout>
__host__
void StereoBM(Tin const *const left, Tin const *const right, Tout *const disparity,
              size_t left_pitch, size_t right_pitch, size_t disparity_pitch,
              int rows, int cols, int max_disparity)
{
    int const tpb = WSIZE;
    dim3 const Dg((cols + 1)/tpb, (rows + 1)/tpb);
    dim3 const Db(tpb, tpb);

    // Copy the images to the device.
    Tin *leftd, *rightd;
    size_t leftd_pitch, rightd_pitch;
    cudaMallocPitch(&leftd, &leftd_pitch, cols * sizeof(Tin), rows);
    cudaMallocPitch(&rightd, &rightd_pitch, cols * sizeof(Tin), rows);
    cudaMemcpy2D(leftd, leftd_pitch, left, left_pitch, cols * sizeof(Tin), rows, cudaMemcpyHostToDevice);
    cudaMemcpy2D(rightd, rightd_pitch, right, right_pitch, cols * sizeof(Tin), rows, cudaMemcpyHostToDevice);

    // Compute the LoG of the left and right images as a simple form of feature
    // extraction.
    Tlog *leftd_log, *rightd_log;
    size_t leftd_log_pitch, rightd_log_pitch;
    cudaMallocPitch(&leftd_log,  &leftd_log_pitch,  cols * sizeof(Tlog), rows);
    cudaMallocPitch(&rightd_log, &rightd_log_pitch, cols * sizeof(Tlog), rows);

    LaplacianOfGaussianD<Tin, Tlog>(
        leftd, leftd_log,
        leftd_pitch, leftd_log_pitch,
        rows, cols
    );
    LaplacianOfGaussianD<Tin, Tlog>(
        rightd, rightd_log,
        rightd_pitch, rightd_log_pitch,
        rows, cols
    );

    // Copy the result back onto the host.
    cudaMemcpy2D(disparity, disparity_pitch, leftd, leftd_pitch,
                 cols * sizeof(Tlog), rows, cudaMemcpyDeviceToHost);

    // Cleanup.
    cudaFree(leftd_log);
    cudaFree(rightd_log);
    cudaFree(leftd);
    cudaFree(rightd);
}

template void StereoBM<uint8_t, int16_t, int32_t, int16_t>(
    uint8_t const *const left, uint8_t const *const right, int16_t *const disparity,
    size_t left_pitch, size_t right_pitch, size_t disparity_pitch,
    int rows, int cols, int max_disparity
);

}
