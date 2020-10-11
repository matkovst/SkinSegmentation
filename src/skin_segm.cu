#pragma once

#include "skin_segm.h"

#define THREADS_PER_BLOCK 16 // max 32

__constant__ float d_Skin_Mus[16][3];
__constant__ float d_Skin_Sigmas[16][3];
__constant__ float d_Skin_Ws[16];
__constant__ float d_Nonskin_Mus[16][3];
__constant__ float d_Nonskin_Sigmas[16][3];
__constant__ float d_Nonskin_Ws[16];


bool cuda_available()
{
    int deviceId = 0;
    cudaError_t err = cudaGetDeviceCount( &deviceId );
    if (err != cudaSuccess) {
        std::cout << "[ERROR] CUDA: No GPU device available. Defaulting to CPU only computation." << std::endl;
        return 0;
    }
    return 1;
}

__device__ float multigauss__gpu(const float x[3], const float mu[3], const float sigma[3], float w)
{
    float det = sigma[0] * sigma[1] * sigma[2];
    if (det == 0)
    {
        return 0.0f;
    }

    float e_coeff = 0;
    float mu_dev[3] = {x[0] - mu[0], x[1] - mu[1], x[2] - mu[2]};
    float tmp[3] = {mu_dev[0] * (1/sigma[0]), mu_dev[1] * (1/sigma[1]), mu_dev[2] * (1/sigma[2])};
    e_coeff = tmp[0] * mu_dev[0] + tmp[1] * mu_dev[1] + tmp[2] * mu_dev[2];
    e_coeff *= -0.5;

    float e = expf(e_coeff);

    float gauss = w * (e / sqrtf(powf(TWOPI, 3) * det));

    return gauss;
}


__device__ float skin_likelihood_gpu(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {d_Skin_Mus[mode][0], d_Skin_Mus[mode][1], d_Skin_Mus[mode][2]};
        const float _sigma[3] = {d_Skin_Sigmas[mode][0], d_Skin_Sigmas[mode][1], d_Skin_Sigmas[mode][2]};
        lhood += multigauss__gpu(pixel, _mean, _sigma, d_Skin_Ws[mode]);
    }
    return lhood;
}


__device__ float nonskin_likelihood_gpu(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {d_Nonskin_Mus[mode][0], d_Nonskin_Mus[mode][1], d_Nonskin_Mus[mode][2]};
        const float _sigma[3] = {d_Nonskin_Sigmas[mode][0], d_Nonskin_Sigmas[mode][1], d_Nonskin_Sigmas[mode][2]};
        lhood += multigauss__gpu(pixel, _mean, _sigma, d_Nonskin_Ws[mode]);
    }
    return lhood;
}

__device__ float segment_skin_gpu(const float pixel[3])
{
    float skin_prob = skin_likelihood_gpu(pixel) * SKIN_PRIOR;
    float nonskin_prob = nonskin_likelihood_gpu(pixel) * NONSKIN_PRIOR;
    float denom = skin_prob + nonskin_prob;
    if (denom == 0)
    {
        return 0.0f;
    }
    else
    {
        return (skin_prob / (denom));
    }
}

__global__ void segment_skin_gpu(float* d_imdata, float* d_out, const int width, const int height)
{
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int w = threadIdx.x + blockIdx.x * blockDim.x;

    int index = w + h * width;
    if (index >= width * height)
    {
        return;
    }

    const float pixel[3] = 
    {
        d_imdata[3 * index + 2], 
        d_imdata[3 * index + 1], 
        d_imdata[3 * index + 0]
    };
    d_out[index] = segment_skin_gpu(pixel);
}

__global__ void segment_skin_pixel_gpu(const float pixel[3], float* out)
{
    float skin_prob = skin_likelihood_gpu(pixel) * 30/100.0f;
    float nonskin_prob = nonskin_likelihood_gpu(pixel) * 70/100.0f;
    float denom = skin_prob + nonskin_prob;
    if (denom == 0)
    {
        *out = 0.f;
    }
    else
    {
        *out = (skin_prob / (denom));
    }
}

int run_tests_gpu(int argc, char** argv)
{
    if (!cuda_available())
    {
        return -1;
    }

    cudaMemcpyToSymbol(d_Skin_Mus, Skin_Mus, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Skin_Sigmas, Skin_Sigmas, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Skin_Ws, Skin_Ws, 16 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Mus, Nonskin_Mus, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Sigmas, Nonskin_Sigmas, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Ws, Nonskin_Ws, 16 * sizeof(float));

    /* Testing posterior prob */
    const float h_pixel[3] = {111, 29, 55};
    float* h_out = (float*)malloc(sizeof(float));
    float* d_pixel;
    float* d_out;
    cudaMalloc((void**)&d_pixel, 3 * sizeof(float));
    cudaMemcpy(d_pixel, h_pixel, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_out, sizeof(float));
    segment_skin_pixel_gpu<<<1,1>>>(d_pixel, d_out);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    bool check = (abs(*h_out - TEST_PIXEL) < 0.001);
    std::cout << "[INFO] Test pixel: " << std::boolalpha << check << std::endl;

    free(h_out);
    cudaFree(d_pixel);
    cudaFree(d_out);

    cudaFree(d_Skin_Mus);
    cudaFree(d_Skin_Sigmas);
    cudaFree(d_Skin_Ws);
    cudaFree(d_Nonskin_Mus);
    cudaFree(d_Nonskin_Sigmas);
    cudaFree(d_Nonskin_Ws);

    return 0;
}


int run_image_gpu(int argc, char** argv)
{
    if (!cuda_available())
    {
        return -1;
    }

    /* Measure GPU time */
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "[INFO] Skin segmentation started on image." << std::endl;

    /* Load image */
    if (argc < 2)
    {
        std::cerr << "[ERROR] Specify img path!" << std::endl;
        return -1;
    }
    
    float resizeRate = 1;
    if (argc > 2) resizeRate = atof(argv[2]);

    std::string imgpath = argv[1];
    cv::Mat img = cv::imread(imgpath);
    if (img.empty())
    {
        std::cerr << "[ERROR] Image loading fucked up!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(0, 0), resizeRate, resizeRate);
    img.convertTo(img, CV_32FC3);

    cudaMemcpyToSymbol(d_Skin_Mus, Skin_Mus, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Skin_Sigmas, Skin_Sigmas, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Skin_Ws, Skin_Ws, 16 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Mus, Nonskin_Mus, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Sigmas, Nonskin_Sigmas, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Ws, Nonskin_Ws, 16 * sizeof(float));

    /* Testing image posterior prob */
    const int width = img.cols;
    const int height = img.rows;
    size_t imsize = 3 * width * height * sizeof(float);
    size_t out_imsize = width * height * sizeof(float);
    float *h_imdata = img.ptr<float>();
    float *h_imout = (float*)malloc(out_imsize);
    float *d_imdata;
    float *d_imout;
    cudaMalloc((void**)&d_imdata, imsize);
    cudaMalloc((void**)&d_imout, out_imsize);
    cudaMemcpy(d_imdata, h_imdata, imsize, cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    int grid_x = floor((width + blockDim.x - 1) / blockDim.x);
    int grid_y = floor((height + blockDim.y - 1) / blockDim.y);
    dim3 gridDim(grid_x, grid_y);

    cudaEventRecord(start, 0);
    segment_skin_gpu<<<gridDim, blockDim>>>(d_imdata, d_imout, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("[INFO] Image elapsed: %3.1f ms\n", elapsedTime);

    cudaMemcpy(h_imout, d_imout, out_imsize, cudaMemcpyDeviceToHost);
    
    /* Save result */
    cv::Mat gpu_out = cv::Mat(height, width, CV_32FC1, h_imout);
    cv::Mat to_save = 255 * gpu_out.clone();
    to_save.convertTo(to_save, CV_8UC1);
    bool good = cv::imwrite("result_gpu.jpg", to_save);
    std::cout << "[INFO] Result image saved as result_gpu.jpg: " << std::boolalpha << good << std::endl;

    /* Cleanup */
    free(h_imout);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_imdata);
    cudaFree(d_imout);

    std::cout << "[INFO] Skin segmentation finished on image." << std::endl;
    return 0;
}

int run_stream_gpu(int argc, char** argv)
{
    if (!cuda_available())
    {
        return -1;
    }

    /* Measure GPU time */
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "[INFO] Skin segmentation started on stream." << std::endl;

    if (argc < 2)
    {
        std::cerr << "[ERROR] Specify input path!" << std::endl;
        return -1;
    }
    std::string inputpath = argv[1];

    float resizeRate = 1;
    if (argc > 2) resizeRate = atof(argv[2]);

    int rotateVideo = 0;
    if (argc > 3) rotateVideo = atoi(argv[3]);


    cudaMemcpyToSymbol(d_Skin_Mus, Skin_Mus, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Skin_Sigmas, Skin_Sigmas, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Skin_Ws, Skin_Ws, 16 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Mus, Nonskin_Mus, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Sigmas, Nonskin_Sigmas, 16 * 3 * sizeof(float));
    cudaMemcpyToSymbol(d_Nonskin_Ws, Nonskin_Ws, 16 * sizeof(float));


    cv::VideoCapture capture;
    if (inputpath == "0") capture.open(0);
    else capture.open(inputpath);
    if( !capture.isOpened() )
    {
        std::cerr << "[ERROR] Could not initialize capturing..." << std::endl;
        return 0;
    }

    cv::Mat frame;
    capture >> frame;
    if( frame.empty() )
    {
        std::cerr << "[ERROR] Could not capture frame..." << std::endl;
        return 0;
    }
    cv::resize(frame, frame, cv::Size(0, 0), resizeRate, resizeRate);
    if (rotateVideo) cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
    frame.convertTo(frame, CV_32FC3);

    const int width = frame.cols;
    const int height = frame.rows;
    size_t imsize = 3 * width * height * sizeof(float);
    size_t out_imsize = width * height * sizeof(float);
    float *h_imdata = frame.ptr<float>();
    float *h_imout = (float*)malloc(out_imsize);
    float *d_imdata;
    float *d_imout;
    cudaMalloc((void**)&d_imdata, imsize);
    cudaMalloc((void**)&d_imout, out_imsize);

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    int grid_x = floor((width + blockDim.x - 1) / blockDim.x);
    int grid_y = floor((height + blockDim.y - 1) / blockDim.y);
    dim3 gridDim(grid_x, grid_y);

    while (true)
    {
        capture >> frame;
        if( frame.empty() )
        {
            break;
        }

        cv::resize(frame, frame, cv::Size(0, 0), resizeRate, resizeRate);
        if (rotateVideo) cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        frame.convertTo(frame, CV_32FC3);

        float *h_imdata = frame.ptr<float>();
        cudaMemcpy(d_imdata, h_imdata, imsize, cudaMemcpyHostToDevice);

        cudaEventRecord(start, 0);
        segment_skin_gpu<<<gridDim, blockDim>>>(d_imdata, d_imout, width, height);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "\r[INFO] Stream elapsed: " << elapsedTime << " ms" << std::flush;

        cudaDeviceSynchronize();
        cudaMemcpy(h_imout, d_imout, out_imsize, cudaMemcpyDeviceToHost);
        cv::Mat out = cv::Mat(height, width, CV_32FC1, h_imout);
        
        // cv::Mat out = cv::Mat::zeros(frame.rows, frame.cols, CV_32FC1);
        // segment_skin_gpu(frame, out);

        cv::imshow(WINDOW_NAME, out);
        char c = (char)cv::waitKey(10);
        if( c == 27 || cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) < 1 )
        {
            break;
        }
    }
    std::cout << std::endl;

    capture.release();
    cv::destroyAllWindows();
    
    /* Cleanup */
    free(h_imout);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_imdata);
    cudaFree(d_imout);

    std::cout << "[INFO] Skin segmentation finished on stream." << std::endl;
    return 0;
}