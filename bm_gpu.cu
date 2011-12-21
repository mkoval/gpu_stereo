#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>
#include "bm_gpu.hpp"

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

template <typename Tin, typename Tker, typename Tout,
          size_t Krows, size_t Kcols>
__global__
void convolve(Tin const *const src, Tout *const dst, Tker const *const ker,
              size_t rows, size_t cols,
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
            value += ker[dr * Kcols + dc] * (Tout)src[r * src_pitch + c];
        }

        dst[r0 * dst_pitch + c0] = value;
    } else {
        dst[r0 * dst_pitch + c0] = 0;
    }
}

template <typename Tin, typename Tout>
__host__
void LaplacianOfGaussian(Tin const *const src, Tout *const dst,
                         size_t rows, size_t cols)
{
    // Copy the input image to the device.
    Tin *srcd;
    size_t srcd_pitch;
    size_t const src_pitch = cols*sizeof(Tin);
    cudaMallocPitch(&srcd, &srcd_pitch, src_pitch, rows);
    cudaMemcpy2D(srcd, srcd_pitch, src, src_pitch, src_pitch, rows,
                 cudaMemcpyHostToDevice);

    // Allocate a buffer for the output on the device.
    Tout *dstd;
    size_t dstd_pitch;
    size_t const dst_pitch = cols*sizeof(Tout);
    cudaMallocPitch(&dstd, &dstd_pitch, cols*sizeof(Tout), rows);

    // Split the image into as many sqrt(WSIZE)*sqrt(WSIZE) blocks as is needed
    // to cover the entire image. This guarantees that there are approximately
    // WSIZE threads per block. Then, solve for the convolution using one thread
    // per pixel.
    int const tpb = (int)floor(sqrt(WSIZE));
    convolve<Tin, int8_t, Tout, KERNEL_LOG_ROWS, KERNEL_LOG_COLS>
            <<<dim3((cols + 1)/tpb, (rows + 1)/tpb), dim3(tpb, tpb)>>>
            (srcd, dstd, (int8_t const*)KERNEL_LOG, rows, cols, src_pitch, dst_pitch);

    // Copy the result back to the host.
    cudaMemcpy2D(dst, dst_pitch, dstd, dstd_pitch, src_pitch, rows,
                 cudaMemcpyDeviceToHost);
    cudaFree(srcd);
    cudaFree(dstd);
}

template void LaplacianOfGaussian(uint8_t const *const src, int16_t *const dst,
                                  size_t rows, size_t cols);

}
