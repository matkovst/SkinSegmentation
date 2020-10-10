#pragma once

#include "skin_segm.h"

float multigauss(const float x[3], const float mu[3], const float sigma[3], float w)
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


float skin_likelihood(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {Skin_Mus[mode][0], Skin_Mus[mode][1], Skin_Mus[mode][2]};
        const float _sigma[3] = {Skin_Sigmas[mode][0], Skin_Sigmas[mode][1], Skin_Sigmas[mode][2]};
        lhood += multigauss(pixel, _mean, _sigma, Skin_Ws[mode]);
    }
    return lhood;
}


float nonskin_likelihood(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {Nonskin_Mus[mode][0], Nonskin_Mus[mode][1], Nonskin_Mus[mode][2]};
        const float _sigma[3] = {Nonskin_Sigmas[mode][0], Nonskin_Sigmas[mode][1], Nonskin_Sigmas[mode][2]};
        lhood += multigauss(pixel, _mean, _sigma, Nonskin_Ws[mode]);
    }
    return lhood;
}


float segment_skin_pixel(const float pixel[3])
{
    float skin_prob = skin_likelihood(pixel) * SKIN_PRIOR;
    float nonskin_prob = nonskin_likelihood(pixel) * NONSKIN_PRIOR;
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


void segment_skin(const cv::Mat& img, cv::Mat& out)
{
    out = cv::Mat(img.rows, img.cols, CV_32FC1);
    int w = img.cols;
    int h = img.rows;
#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int i = 0; i < w * h; i++)
    {
        cv::Vec3f pixel = img.at<cv::Vec3f>(i / w, i % w);
        const float fpixel[3] = {pixel[0], pixel[1], pixel[2]};
        out.at<float>(i / w, i % w) = segment_skin_pixel(fpixel);
    }
}


void segment_skin_fast(const cv::Mat& img, cv::Mat& out)
{
    out = cv::Mat(img.rows, img.cols, CV_32FC1);
    int w = img.cols;
    int h = img.rows;
    cv::parallel_for_(cv::Range(0, h * w), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; i++)
        {
            cv::Vec3f pixel = img.at<cv::Vec3f>(i / w, i % w);
            const float fpixel[3] = {pixel[0], pixel[1], pixel[2]};
            out.at<float>(i / w, i % w) = segment_skin_pixel(fpixel);
        }
    }, OPENCV_THREADS);
}


int test_stream(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Specify input path!" << std::endl;
        return -1;
    }
    std::string inputpath = argv[1];

    float resizeRate = 1;
    if (argc > 2) resizeRate = atof(argv[2]);

    int rotateVideo = 0;
    if (argc > 3) rotateVideo = atoi(argv[3]);

    cv::VideoCapture capture;
    if (inputpath == "0") capture.open(0);
    else capture.open(inputpath);
    if( !capture.isOpened() )
    {
        std::cout << "Could not initialize capturing..." << std::endl;
        return 0;
    }

    cv::Mat frame;
    capture >> frame;
    if( frame.empty() )
    {
        std::cout << "Could not capture frame..." << std::endl;
        return 0;
    }
    cv::resize(frame, frame, cv::Size(0, 0), resizeRate, resizeRate);
    if (rotateVideo) cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

    cv::VideoWriter writer("out.mkv", cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(frame.cols, frame.rows), true);

    while (true)
    {
        capture >> frame;
        if( frame.empty() )
        {
            break;
        }
        writer.write(frame);

        cv::resize(frame, frame, cv::Size(0, 0), resizeRate, resizeRate, cv::INTER_AREA);
        if (rotateVideo) cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        frame.convertTo(frame, CV_32FC3);
        
        clock_t begin = clock();
        cv::Mat out;
        segment_skin_fast(frame, out);
        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << '\r' << "[INFO] Stream Elapsed: " << elapsed << std::flush;

        cv::resize(out, out, cv::Size(0, 0), 1/resizeRate, 1/resizeRate, cv::INTER_AREA);
        cv::imshow(WINDOW_NAME, out);
        char c = (char)cv::waitKey(10);
        if( c == 27 )
        {
            break;
        }
    }
    std::cout << std::endl;
    
    capture.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}

int run_tests_cpu(int argc, char** argv)
{
    /* Testing posterior prob */
    const float fpixel[3] = {111, 29, 55};
    float test_posterior = segment_skin_pixel(fpixel);
    bool check = (abs(test_posterior - TEST_PIXEL) < 0.001);
    std::cout << "[INFO] Test pixel: " << std::boolalpha << check << std::endl;

    return 0;
}

int run_image_cpu(int argc, char** argv)
{
    std::cout << "[INFO] Skin segmentation started on image." << std::endl;

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
        std::cerr << "[ERROR] Image loading f*cked up!" << std::endl;
        return -1;
    }
    else
    {
        cv::resize(img, img, cv::Size(0, 0), resizeRate, resizeRate);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3);
    }
    cv::Mat res = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int col = 0; col < img.cols; col++)
    {
        for (int row = 0; row < img.rows; row++)
        {
            cv::Vec3f pixel = img.at<cv::Vec3f>(row, col);
            const float fpixel[3] = {pixel[0], pixel[1], pixel[2]};
            res.at<float>(row, col) = segment_skin_pixel(fpixel);
        }
    }

    /* Save result */
    cv::Mat to_save = 255 * res.clone();
    to_save.convertTo(to_save, CV_8UC1);
    bool good = cv::imwrite("result.jpg", to_save);
    std::cout << "[INFO] Result image saved as result.jpg: " << std::boolalpha << good << std::endl;

    std::cout << "[INFO] Skin segmentation finished on image." << std::endl;
    return 0;
}

int run_stream_cpu(int argc, char** argv)
{
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

    while (true)
    {
        capture >> frame;
        if( frame.empty() )
        {
            break;
        }

        cv::resize(frame, frame, cv::Size(0, 0), resizeRate, resizeRate, cv::INTER_AREA);
        if (rotateVideo) cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        frame.convertTo(frame, CV_32FC3);
        
        clock_t begin = clock();

        cv::Mat out;
        segment_skin_fast(frame, out);

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;
        // std::cout << "[INFO] Stream elapsed: " << elapsed << std::endl;

        // cv::resize(out, out, cv::Size(0, 0), 1/resizeRate, 1/resizeRate, cv::INTER_AREA);
        cv::imshow(WINDOW_NAME, out);
        char c = (char)cv::waitKey(10);
        if( c == 27 || cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) < 1 )
        {
            break;
        }
    }
    capture.release();
    cv::destroyAllWindows();

    std::cout << "[INFO] Skin segmentation finished on stream." << std::endl;
    return 0;
}