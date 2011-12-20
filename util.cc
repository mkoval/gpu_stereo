#include <stdint.h>
#include <limits>
#include "util.h"

using cv::DataType;
using cv::Mat;
using cv::Mat_;
using cv::Scalar;
using std::numeric_limits;

template <typename T>
static T abs(T x)
{
    return (x >= 0) ? x : -x;
}

template <typename Tin, typename Tker, typename Tout, int Krows, int Kcols>
static void convolve(Mat const &src, Mat &dst, Mat const &ker)
{
    CV_Assert(src.rows >= ker.rows && src.cols >= ker.cols);
    CV_Assert(ker.rows == Krows    && ker.cols == Kcols);
    CV_Assert(src.type() == CV_MAKETYPE(DataType<Tin>::depth, 1)
           && ker.type() == CV_MAKETYPE(DataType<Tker>::depth, 1));

    dst.create(src.rows, src.cols, CV_MAKETYPE(DataType<Tout>::depth, 1));

    for (int r0 = Krows/2; r0 < src.rows - Krows/2; r0++) {
        Tout *const dst_row = dst.ptr<Tout>(r0);

        for (int c0 = Kcols/2; c0 < src.cols - Kcols/2; c0++) {
            for (int dr = 0; dr < Krows; dr++) {
                int const r = r0 + dr - Krows/2;
                Tin  const *const src_row = src.ptr<Tin>(r);
                Tker const *const ker_row = ker.ptr<Tker>(dr);

                for (int dc = 0; dc < Kcols; dc++) {
                    int const c = c0 + dc - Kcols/2;
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
    convolve<Tin, int8_t, Tout, 9, 9>(src, dst, ker);
}

template <typename Tin, typename Tlog, typename Terr, typename Tout, int Wrows, int Wcols, int D>
static void MatchBM(Mat const &left, Mat const &right, Mat &disparity)
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
    disparity.setTo(0);

    for (int r0 = Wrows/2; r0 < left.rows - Wrows/2; r0++) {
        Tout *const disparity_row = disparity.ptr<Tout>(r0);

        for (int c0 = Wcols/2 + D; c0 < left.cols - Wcols/2; c0++) {
            Terr best_error     = numeric_limits<Terr>::max();
            Tout best_disparity = 0;

            for (Tout d = 0; d <= D; d++) {
                Terr error = 0;

                for (int dr = -Wrows/2; dr < Wrows/2; dr++) {
                    int const r = r0 + dr;
                    Tlog const *const left_row  = left_log.ptr<Tlog>(r);
                    Tlog const *const right_row = right_log.ptr<Tlog>(r);

                    for (int dc = -Wcols/2; dc < Wcols/2; dc++) {
                        int const c_left  = c0 + dc;
                        int const c_right = c0 + dc - d;
                        error += abs<Terr>((Terr)left_row[c_left] - (Terr)right_row[c_right]);
                    }
                }

                if (error < best_error) {
                    best_error     = error;
                    best_disparity = d;
                }
            }
            disparity_row[c0] = best_disparity;
        }
    }
}

void MatchBM(Mat const &left, Mat const &right, Mat &disparity)
{
    MatchBM<uint8_t, int16_t, int32_t, int32_t, 25, 25, 64>(left, right, disparity);
}
