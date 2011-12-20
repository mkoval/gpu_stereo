#include <iostream>
#include <string>
#include <sstream>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#pragma clang diagnostic pop

#include "util.h"

using cv::Mat;
using cv::Range;
using cv::Size;
using cv::StereoBM;
using cv::gpu::GpuMat;
using cv::gpu::StereoBM_GPU;

#if 0
static int const max_disparity  = StereoBM_GPU::DEFAULT_NDISP;
static int const window_size    = StereoBM_GPU::DEFAULT_WINSZ;
static int const texture_thresh = 0;

static void match_opencv_cpu(Mat const &left, Mat const &right, Mat &disparity)
{
    StereoBM matcher(CV_STEREO_BM_BASIC, max_disparity, window_size);

    disparity.create(left.rows, left.cols, CV_32FC1);
    matcher(left, right, disparity);
}

static void match_opencv_gpu(Mat const &left, Mat const &right, Mat &disparity)
{
    StereoBM_GPU matcher(CV_STEREO_BM_BASIC, max_disparity, window_size);
    matcher.avergeTexThreshold = texture_thresh;

    GpuMat left_gpu(left), right_gpu(right), disparity_gpu;
    matcher(left_gpu, right_gpu, disparity_gpu);
    disparity = disparity_gpu;
}

static void match_cpu(Mat const &left, Mat const &right, Mat &disparity)
{
    static const Size blur_size(3, 3);
    static const int win_size = 25;
    static const int log_size = 1;
    static const int max_disp = 64;
    int win_half = win_size / 2;

    disparity.create(left.rows, left.cols, CV_32FC1);
    disparity.setTo(0.0f);

    // Compute the Laplacian of the Gaussian (LoG) of both images.
    Mat left_gaussian, left_log;
    cv::GaussianBlur(left, left_gaussian, blur_size, 0.0, 0.0);
    cv::Laplacian(left_gaussian, left_log, CV_64FC1, log_size);

    Mat right_gaussian, right_log;
    cv::GaussianBlur(right, right_gaussian, blur_size, 0.0, 0.0);
    cv::Laplacian(right_gaussian, right_log, CV_64FC1, log_size);

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

int main(int argc, char **argv)
{
    if (argc <= 4) {
        std::cerr << "err: incorrect number of arguments\n"
                  << "usage: ./stereo <left image> <right image> <algo> <repeats>\n";
        return 1;
    }

    Mat const left  = cv::imread(argv[1], 0);
    Mat const right = cv::imread(argv[2], 0);
    std::string const algo = argv[3];

    int repeats;
    std::stringstream ss(argv[4]);
    ss >> repeats;

    if (left.rows != right.rows || left.cols != right.cols) {
        std::cerr << "err: both images must be the same size\n";
        return 1;
    }

    Mat disparity;
    LaplacianOfGaussian(left, disparity);

    Mat disparity_norm;
    cv::normalize(disparity, disparity_norm,  0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("disparity.png", disparity_norm);
    return 0;
}
