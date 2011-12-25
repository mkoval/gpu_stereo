#ifndef BM_GPU_HPP_
#define BM_GPU_HPP_

#include <stdint.h>

namespace gpu {

template <typename Tin, typename Tout>
void LaplacianOfGaussian(Tin const *const src, Tout *const dst,
                         size_t src_pitch, size_t dst_pitch,
                         int rows, int cols);

template <typename Tin, typename Tlog, typename Terr, typename Tout>
void StereoBM(Tin const *const left, Tin const *const right, Tout *const dst,
              size_t left_pitch, size_t right_pitch, size_t dst_pitch,
              int rows, int cols);

}

#endif
