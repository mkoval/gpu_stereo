#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

using cv::Mat;
using cv::Range;
using cv::gpu::GpuMat;

static int const max_disparity = 64;
static int const window_size   = 21;

int main(int argc, char **argv)
{
    cv::StereoBM bm_opencv_cpu(CV_STEREO_BM_BASIC, max_disparity, window_size);
    cv::gpu::StereoBM_GPU bm_opencv_gpu(CV_STEREO_BM_BASIC, max_disparity, window_size);

    if (argc < 1) {
        std::cerr << "err: incorrect number of arguments"
                  << "usage: ./stereo"
                  << std::endl;
        return 1;
    }

    Mat const left  = cv::imread("images/left.png", 0);
    Mat const right = cv::imread("images/right.png", 0);

#if 0
    Mat disparity(left.rows, left.cols, CV_32FC1);
    bm_opencv_cpu(left, right, disparity);
#else
    GpuMat left_gpu, right_gpu;
    left_gpu.upload(left);
    right_gpu.upload(right);

    GpuMat disparity_gpu;
    Mat disparity;
    bm_opencv_gpu(left_gpu, right_gpu, disparity_gpu);
    disparity_gpu.download(disparity);
#endif

    Mat render(left.rows, 3*left.cols, CV_8UC1);
    Mat render_left  = render(Range::all(), Range(0, left.cols));
    Mat render_right = render(Range::all(), Range(left.cols, 2*left.cols));
    Mat render_disp  = render(Range::all(), Range(2*left.cols, 3*left.cols));
    left.copyTo(render_left);
    right.copyTo(render_right);
    cv::normalize(disparity, render_disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("Stereo Pair", render);
    cv::waitKey();

    return 0;
}
