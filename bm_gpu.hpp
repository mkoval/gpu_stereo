#ifndef BM_GPU_HPP_
#define BM_GPU_HPP_

#include <stdint.h>

namespace gpu {

template <typename Tin, typename Tout>
void LaplacianOfGaussian(Tin const *const src, Tout *const dst,
                         size_t src_pitch, size_t dst_pitch,
                         size_t rows, size_t cols);

}

#endif
