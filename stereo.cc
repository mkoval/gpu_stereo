#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

using cv::Mat;
using cv::Range;
using cv::StereoBM;
using cv::gpu::GpuMat;
using cv::gpu::StereoBM_GPU;

static int const max_disparity  = StereoBM_GPU::DEFAULT_NDISP;
static int const window_size    = StereoBM_GPU::DEFAULT_WINSZ;
static int const texture_thresh = 0;

void match_opencv_cpu(Mat const &left, Mat const &right, Mat &disparity)
{
    static StereoBM matcher(CV_STEREO_BM_BASIC, max_disparity, window_size);

    disparity.create(left.rows, left.cols, CV_32FC1);
    matcher(left, right, disparity);
}

void match_opencv_gpu(Mat const &left, Mat const &right, Mat &disparity)
{
    static StereoBM_GPU matcher(CV_STEREO_BM_BASIC, max_disparity, window_size);
    matcher.avergeTexThreshold = texture_thresh;

    GpuMat left_gpu(left), right_gpu(right), disparity_gpu;
    matcher(left_gpu, right_gpu, disparity_gpu);
    disparity = disparity_gpu;
}

int main(int argc, char **argv)
{

    if (argc < 3) {
        std::cerr << "err: incorrect number of arguments"         << std::endl
                  << "usage: ./stereo <left image> <right image>" << std::endl;
        return 1;
    }

    Mat const left  = cv::imread(argv[1], 0);
    Mat const right = cv::imread(argv[2], 0);

    if (left.rows != right.rows || left.cols != right.cols) {
        std::cerr << "err: both images must be the same size" << std::endl;
        return 1;
    }

    // OpenCV CPU Stereo
    Mat disparity_cpu;
    match_opencv_cpu(left, right, disparity_cpu);

    // OpenCV GPU Stereo
    Mat disparity_gpu;
    match_opencv_gpu(left, right, disparity_gpu);

    // Render the 
    Mat render(left.rows, 3*left.cols, CV_8UC1);
    Mat render_left = render(Range::all(), Range(0, left.cols));
    Mat render_cpu  = render(Range::all(), Range(left.cols, 2*left.cols));
    Mat render_gpu  = render(Range::all(), Range(2*left.cols, 3*left.cols));
    left.copyTo(render_left);
    cv::normalize(disparity_cpu, render_cpu, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(disparity_gpu, render_gpu, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("Stereo Pair", render);
    cv::waitKey();

    return 0;
}
