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
    MatchBM(left, right, disparity);

    Mat disparity_norm;
    cv::normalize(disparity, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("disparity.png", disparity_norm);
    return 0;
}
