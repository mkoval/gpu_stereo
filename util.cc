#include <stdint.h>
#include "util.h"

#include <iostream>

using cv::Mat;
using cv::Scalar;

template <typename Tin, typename Tout, int Krows, int Kcols>
static void convolve(Mat const &src, Mat &dst, Tout const ker[Krows][Kcols])
{
    for (int r0 = Krows/2; r0 < src.rows - Krows/2; r0++) {
        Tout *dst_row = dst.ptr<Tout>(r0);

        for (int c0 = Kcols/2; c0 < src.cols - Kcols/2; c0++) {
            for (int dr = 0; dr < Krows; dr++) {
                int const r = r0 + dr - Krows/2;
                Tin const *const src_row = src.ptr<Tin>(r);

                for (int dc = 0; dc < Kcols; dc++) {
                    int const c = c0 + dc - Kcols/2;
                    dst_row[c] += ker[dr][dc] * src_row[c];
                }
            }
        }
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

#if 0
template <int Wrows, int Wcols, int D>
static void MatchBM(Mat const &left, Mat const &right, Mat &disparity)
{
    Mat left_log, right_log;
    LaplacianOfGaussian(left, left_log);
    LaplacianOfGaussian(right, right_log);

    disparity.create(left.rows, left.cols, CV_16U);
    disparity.setTo(0.0f);

    for (int r0 = Wrows/2; r0 < left.rows - Wrows/2; r0++)
    for (int c0 = Wcols/2; c0 < left.cols - Wcols/2; c0++) {
        uint16_t best_error     = INT16_MAX;
        uint16_t best_disparity = 0;

        for (int disparity = 0; disparity <= D; disparity++) {
            uint16_t error = 0;

            for (int dr = Wrows; dr < Wrows; dr++)
            for (int dc = Wcols; dc < Wcols; dc++) {
                int const r = r0 + dr - Wrows / 2;
                int const c = c0 + dc - Wcols / 2;
                uint8_t const left_px  = left.data[r*left.step[0] + c*left.step[1]];
                uint8_t const right_px = right.data[r*right.step[0] + c*right.step[1]];
                error += abs(left_px - right_px);
            }

            if (error < best_error) {
                best_error     = error;
                best_disparity = disparity;
            }
        }

        disparity->
    }

    for (int row = win_half; row < left.rows - win_half; row++)
    for (int col = max_disp + win_half; col < left.cols - win_half; col++) {
        double best_sad = INFINITY;
        int best_offset = 0;

        Range patch_rows(row - win_half, row + win_half);
        Range patch_cols_left(col - win_half, col + win_half);
        Mat patch_left = left_log(patch_rows, patch_cols_left);

        // Features in the right image will be left of the corresponding
        // feature in the left image, so we only need to consider candidate
        // pixels to the left of our current pixel.
        for (int offset = 0; offset < max_disp; offset++) {
            Range patch_cols_right(col - offset - win_half, col - offset + win_half);
            Mat patch_right = right_log(patch_rows, patch_cols_right);

            double sad = cv::norm(patch_left, patch_right, cv::NORM_L2);
            if (sad < best_sad) {
                best_offset = offset;
                best_sad = sad;
            } 
        }
        disparity.at<float>(row, col) = best_offset;
    }
}
#endif
