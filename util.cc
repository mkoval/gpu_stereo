#include <algorithm>
#include <limits>
#include <vector>
#include <stdint.h>
#include "util.h"

using cv::DataType;
using cv::Mat;
using cv::Mat_;
using cv::Scalar;
using std::numeric_limits;
using std::vector;

static int DISPARITY_NONE = 0;

template <typename T>
static T abs(T x)
{
    return (x >= 0) ? x : -x;
}

template <typename Tin, typename Tker, typename Tout>
static void convolve(Mat const &src, Mat &dst, Mat const &ker)
{
    CV_Assert(src.rows >= ker.rows && src.cols >= ker.cols);
    CV_Assert(src.type() == CV_MAKETYPE(DataType<Tin>::depth, 1)
           && ker.type() == CV_MAKETYPE(DataType<Tker>::depth, 1));

    dst.create(src.rows, src.cols, CV_MAKETYPE(DataType<Tout>::depth, 1));

    for (int r0 = ker.rows/2; r0 < src.rows - ker.rows/2; r0++) {
        Tout *const dst_row = dst.ptr<Tout>(r0);

        for (int c0 = ker.cols/2; c0 < src.cols - ker.cols/2; c0++) {
            for (int dr = 0; dr < ker.rows; dr++) {
                int const r = r0 + dr - ker.rows/2;
                Tin  const *const src_row = src.ptr<Tin>(r);
                Tker const *const ker_row = ker.ptr<Tker>(dr);

                for (int dc = 0; dc < ker.cols; dc++) {
                    int const c = c0 + dc - ker.cols/2;
                    dst_row[c] += ker_row[dc] * src_row[c];
                }
            }
        }
    }
}

template <typename Tin, typename Tout>
static void LaplacianOfGaussian(Mat const &src, Mat &dst)
{
    static Mat const ker = (Mat_<int8_t>(9, 9) <<
        0, 1, 1,   2,   2,   2, 1, 1, 0,
        1, 2, 4,   5,   5,   5, 4, 2, 1,
        1, 4, 5,   3,   0,   3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5,   3,   0,   3, 5, 4, 1,
        1, 2, 4,   5,   5,   5, 4, 2, 1,
        0, 1, 1,   2,   2,   2, 1, 1, 0
    );
    convolve<Tin, int8_t, Tout>(src, dst, ker);
}

template <typename Tin, typename Tlog, typename Terr, typename Tout>
static void MatchBM(Mat const &left, Mat const &right, Mat &disparity,
                    int Wrows, int Wcols, int D)
{
    CV_Assert(Wrows > 0 && Wcols > 0 && D >= 0);
    CV_Assert(left.rows == right.rows && left.rows >= Wrows
           && left.cols == right.cols && left.cols >= Wcols);
    CV_Assert(left.type()  == CV_MAKETYPE(DataType<Tin>::depth, 1)
           && right.type() == CV_MAKETYPE(DataType<Tin>::depth, 1));

    Mat left_log, right_log;
    LaplacianOfGaussian<Tin, Tlog>(left, left_log);
    LaplacianOfGaussian<Tin, Tlog>(right, right_log);

    disparity.create(left.rows, left.cols, CV_MAKETYPE(DataType<Tout>::depth, 1));

    // Precompute a vertical integral image of the SAD kernel along each row in
    // O(w*h*D) + O(w*h) complexity. This reduces the complexity of the
    // disparity calculation from O(w*h*D^2) to O(w*h*D).
    vector<Mat> precomp_sad(D + 1);
    for (int d = 0; d <= D; d++) {
        Mat sad(left.rows, left.cols, CV_MAKETYPE(DataType<Terr>::depth, 1),
                Scalar::all(0));

        // Compute the SAD image.
        for (int r0 = 0; r0 < left.rows; r0++) {
            Tin const *const left_row  = left.ptr<Tin>(r0);
            Tin const *const right_row = right.ptr<Tin>(r0);
            Terr *const sad_row = sad.ptr<Terr>(r0);

            for (int c0 = Wcols/2; c0 < left.cols - Wcols/2; c0++)
            for (int c = c0 - Wcols/2; c < c0 + Wcols/2; c++) {
                sad_row[c0] += abs<Terr>((Terr)left_row[c] - (Terr)right_row[c - d]);
            }
        }

        // Sum along columns to calculate the integral image.
        Terr const *prev_row = sad.ptr<Terr>(0);
        for (int r = 1; r < left.rows; r++) {
            Terr *const cur_row = sad.ptr<Terr>(r);

            for (int c = 0; c < left.cols; c++) {
                cur_row[c] += prev_row[c];
            }
            prev_row = cur_row;
        }

        precomp_sad[d] = sad;
    }

    for (int r0 = Wrows/2; r0 < left.rows - Wrows/2; r0++) {
        Tout *const disparity_row = disparity.ptr<Tout>(r0);

        // Initialize the borders of the disparity image. These pixels are
        // guaranteed to have no correspondences.
        for (int c0 = 0; c0 < Wcols/2; c0++) {
            disparity_row[c0] = DISPARITY_NONE;
        }
        for (int c0 = left.cols - Wcols/2; c0 < left.cols; c0++) {
            disparity_row[c0] = DISPARITY_NONE;
        }

        for (int c0 = Wcols/2; c0 < left.cols - Wcols/2; c0++) {
            Terr best_error     = numeric_limits<Terr>::max();
            Tout best_disparity = 0;

            // Search for the pixel (r, c) in the right image that minimizes
            // the sum of absolute difference (SAD) the window surrounding (r0,
            // c0) in the left image.
            int const Drow = std::min(D, c0 - Wcols/2);
            for (Tout d = 0; d <= Drow; d++) {
                // Add the pre-computed partial row-SADs to calculate the total
                // SAD in the window.
                Mat const sad = precomp_sad[d];
                Terr error = sad.at<Terr>(r0 + Wrows/2, c0) - sad.at<Terr>(r0 - Wrows/2, c0);

                if (error < best_error) {
                    best_error     = error;
                    best_disparity = d;
                }
            }
            disparity_row[c0] = best_disparity;
        }
    }
}

void MatchBM(Mat const &left, Mat const &right, Mat &disparity,
             int ndisparities, int window_size)
{
    MatchBM<uint8_t, int16_t, int32_t, int32_t>(left, right, disparity,
                                                window_size, window_size,
                                                ndisparities);
}
