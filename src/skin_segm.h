#pragma once

#include <stdio.h>
#include <iostream>
#include "omp.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "weights.h"

#define WINDOW_NAME "Skin segmentation demo"

float multigauss(const float x[3], const float mu[3], const float sigma[3], float w);
float skin_likelihood(const float pixel[3]);
float nonskin_likelihood(const float pixel[3]);
void segment_skin(const cv::Mat& img, cv::Mat& out);
void segment_skin_fast(const cv::Mat& img, cv::Mat& out);
float segment_skin_pixel(const float pixel[3]);

int run_tests_cpu(int argc, char** argv);
int run_image_cpu(int argc, char** argv);
int run_stream_cpu(int argc, char** argv);

#ifdef WITH_CUDA // GPU

#include <cuda.h>
#include <cuda_runtime.h>

bool cuda_available();

__device__ float multigauss_gpu(const float x[3], const float mu[3], const float sigma[3], float w);
__device__ float skin_likelihood_gpu(const float pixel[3]);
__device__ float nonskin_likelihood_gpu(const float pixel[3]);
__device__ float segment_skin_gpu(const float pixel[3]);
__global__ void segment_skin_gpu(float* d_imdata, float* d_out, const int width, const int height);
__global__ void segment_skin_pixel_gpu(const float pixel[3], float* out);

int run_tests_gpu(int argc, char** argv);
int run_image_gpu(int argc, char** argv);
int run_stream_gpu(int argc, char** argv);

#endif // GPU