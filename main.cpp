#include <iostream>
#include <cmath>
#include "omp.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "src/skin_segm.h"
#include "src/weights.h"

int main(int argc, char** argv)
{
    cv::setNumThreads(OPENCV_THREADS);

    std::cout << "[INFO] Program started ..." << std::endl;

#ifndef WITH_CUDA // CPU

    std::cout << "[INFO] Use CPU" << std::endl;

    int tests_good = run_tests_cpu(argc, argv);
    // int img_good = run_image_cpu(argc, argv);
    int stream_good = run_stream_cpu(argc, argv);

#else // GPU

    std::cout << "[INFO] Use GPU" << std::endl;

    int tests_good = run_tests_gpu(argc, argv);
    // int img_good = run_image_gpu(argc, argv);
    int img_good = run_stream_gpu(argc, argv);

#endif // GPU


    std::cout << "[INFO] Program finished." << std::endl;
    return 0;
}
