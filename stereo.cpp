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

    // Time the algorithm over a large number of iterations.
    Mat disparity;
    disparity.create(left.rows, left.cols, CV_16SC1);

#if 0
    gpu::StereoBM<uint8_t, int16_t, int32_t, int16_t>(
        left.ptr<uint8_t>(0), right.ptr<uint8_t>(0), disparity.ptr<int16_t>(0),
        left.step[0], right.step[0], disparity.step[0],
        left.rows, left.cols, 64
    );
#endif
    gpu::LaplacianOfGaussian(
        left.data, (int16_t *)disparity.data,
        left.cols * sizeof(uint8_t), left.cols * sizeof(int16_t),
        left.rows, left.cols
    );

    Mat disparity_norm;
    cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("disparity.png", disparity_norm);
    return 0;
}