#include <iostream>
#include <string>
#include <sstream>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "bm_cpu.hpp"
#include "bm_gpu.hpp"

using cv::Mat;
using cv::Range;
using cv::Size;
using cv::StereoBM;
using cv::gpu::GpuMat;
using cv::gpu::StereoBM_GPU;
using std::cerr;
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
    if (argc <= 4) {
        std::cerr << "err: incorrect number of arguments\n"
                  << "usage: ./stereo <left image> <right image> <repeats>\n";
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

    cv::Mat const kernel = (cv::Mat_<int8_t>(9, 9) <<
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

    cv::gpu::GpuMat gpu_left(left), gpu_kernel(kernel), gpu_disp;
    gpu::convolve(gpu_left, gpu_kernel, gpu_disp);
    cv::Mat disparity = gpu_disp;

    Mat disparity_norm;
    cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("disparity.png", disparity_norm);
    return 0;
}
