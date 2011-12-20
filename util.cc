#include <stdint.h>
#include "util.h"

#include <iostream>

using cv::Mat;
using cv::Scalar;

template <typename Tin, typename Tout, int Krows, int Kcols>
static void convolve(Mat const &src, Mat &dst, Tout const ker[Krows][Kcols])
{
    for (int r0 = Krows/2; r0 < src.rows - Krows/2; r0++)
    for (int c0 = Kcols/2; c0 < src.cols - Kcols/2; c0++) {
        Tout dst_px = 0;

        for (int dr = 0; dr < Krows; dr++)
        for (int dc = 0; dc < Kcols; dc++) {
            int const r = r0 + dr - Krows/2;
            int const c = c0 + dc - Kcols/2;
            dst_px += ker[dr][dc] * (Tin)src.data[r*src.step[0] + c*src.step[1]];
        }

        memcpy(dst.data + r0 * dst.step[0] + c0 * dst.step[1], &dst_px, sizeof(Tout));
    }
}

void LaplacianOfGaussian(Mat const &src, Mat &dst)
{
    static int16_t const ker[9][9] = {
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
    dst.create(src.rows, src.cols, CV_16SC1);
    convolve<uint8_t, int16_t, 9, 9>(src, dst, ker);
}
