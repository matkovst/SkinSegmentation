#define _USE_MATH_DEFINES
#include <cmath>

#include "skin_segm.h"

float bgrMultigauss(const float x[3], const float* mu, const float* precision, float precompCoeff)
{
    const float meanDiff2[3] = {powf(x[0] - mu[2], 2), powf(x[1] - mu[1], 2), powf(x[2] - mu[0], 2)};
    const float scaledMeanDiff[3] = {meanDiff2[0] * precision[2], meanDiff2[1] * precision[1], meanDiff2[2] * precision[0]};
    const float expArg = -0.5f * (scaledMeanDiff[0] + scaledMeanDiff[1] + scaledMeanDiff[2]);

    const float e = std::exp(expArg);
    const float prob = e / precompCoeff;

    return prob;
}


float skinLikelihood(const float pixel[3])
{
    float lhood = 0.0f;
    for (int mode = 0; mode < 16; ++mode)
        lhood += bgrMultigauss(
            pixel, Skin_Mus[mode], PrecomputedSkin_Precisions[mode], PrecomputedSkin_GaussCoeff[mode]);
    return lhood;
}


float nonskinLikelihood(const float pixel[3])
{
    float lhood = 0.0f;
    for (int mode = 0; mode < 16; ++mode)
        lhood += bgrMultigauss(
            pixel, Nonskin_Mus[mode], PrecomputedNonskin_Precisions[mode], PrecomputedNonskin_GaussCoeff[mode]);
    return lhood;
}


float segmentSkinPixel(const float pixel[3])
{
    const float skinProb = skinLikelihood(pixel) * SkinPrior;
    const float nonskinProb = nonskinLikelihood(pixel) * NonskinPrior;
    const float denom = skinProb + nonskinProb;
    if (denom == 0.0f)
        return 0.0f;
    else
        return skinProb / denom;
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
        out.at<float>(i / w, i % w) = segmentSkinPixel(fpixel);
    }
}


void segmentSkinFast(const cv::Mat& img, cv::Mat& out)
{
    out = cv::Mat(img.rows, img.cols, CV_32FC1);
    int w = img.cols;
    int h = img.rows;
    cv::parallel_for_(cv::Range(0, h * w), [&](const cv::Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            auto pixel = cv::Vec3f(img.at<cv::Vec3b>(i / w, i % w));
            const float fpixel[3] = {pixel[0], pixel[1], pixel[2]};
            out.at<float>(i / w, i % w) = segmentSkinPixel(fpixel);
        }
    }, OPENCV_THREADS);
}


void _experimental_segment_skin_opencv(const cv::Mat& img, cv::Mat& out)
{
    out = cv::Mat(img.rows, img.cols, CV_32FC1);
    int w = img.cols;
    int h = img.rows;

    // Compute skin likelihood
    cv::Mat skinLikelihood;
    for (int mode = 0; mode < 16; ++mode)
    {
        cv::Mat meanDiff;
        cv::subtract(
            img, cv::Scalar(Skin_Mus[mode][0], Skin_Mus[mode][1], Skin_Mus[mode][2]), meanDiff, 
            cv::noArray(), CV_32FC3);
        cv::Mat scaledMeanDiff;
        cv::multiply(
            meanDiff, 
            cv::Scalar(PrecomputedSkin_Precisions[mode][0], PrecomputedSkin_Precisions[mode][1], PrecomputedSkin_Precisions[mode][2]), 
            scaledMeanDiff, 
            1.0, CV_32FC3);
        cv::Mat meanDiff2;
        cv::multiply(scaledMeanDiff, meanDiff, meanDiff2, 1.0, CV_32FC3);
        cv::Mat expArg;
        cv::transform(meanDiff2, expArg, cv::Matx13f(-0.5f, -0.5f, -0.5f));

        cv::Mat e;
        cv::exp(expArg, e);
        cv::multiply(e, 1.0f / PrecomputedSkin_GaussCoeff[mode], skinLikelihood, SkinPrior, CV_32F);
    }

    // Compute non-skin likelihood
    cv::Mat nonskinLikelihood;
    for (int mode = 0; mode < 16; ++mode)
    {
        cv::Mat meanDiff;
        cv::subtract(
            img, cv::Scalar(Nonskin_Mus[mode][0], Nonskin_Mus[mode][1], Nonskin_Mus[mode][2]), meanDiff, 
            cv::noArray(), CV_32FC3);
        cv::Mat scaledMeanDiff;
        cv::multiply(
            meanDiff, 
            cv::Scalar(PrecomputedNonskin_Precisions[mode][0], PrecomputedNonskin_Precisions[mode][1], PrecomputedNonskin_Precisions[mode][2]), 
            scaledMeanDiff, 
            1.0, CV_32FC3);
        cv::Mat meanDiff2;
        cv::multiply(scaledMeanDiff, meanDiff, meanDiff2, 1.0, CV_32FC3);
        cv::Mat expArg;
        cv::transform(meanDiff2, expArg, cv::Matx13f(-0.5f, -0.5f, -0.5f));

        cv::Mat e;
        cv::exp(expArg, e);
        cv::multiply(e, 1.0f / PrecomputedNonskin_GaussCoeff[mode], nonskinLikelihood, NonskinPrior, CV_32F);
    }

    cv::Mat denom = skinLikelihood + nonskinLikelihood;
    out = skinLikelihood / denom;
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
        if (rotateVideo)
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        
        clock_t begin = clock();
        cv::Mat out;
        segmentSkinFast(frame, out);
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
    const float fpixel[3] = {55.0f, 29.0f, 111.0f};
    float test_posterior = segmentSkinPixel(fpixel);
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
            res.at<float>(row, col) = segmentSkinPixel(fpixel);
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
        if (rotateVideo)
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        
        clock_t begin = clock();

        cv::Mat out;
        segmentSkinFast(frame, out);
        // _experimental_segment_skin_opencv(frame, out);

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;
        // std::cout << "[INFO] Stream elapsed: " << elapsed << std::endl;

        // cv::resize(out, out, cv::Size(0, 0), 1/resizeRate, 1/resizeRate, cv::INTER_AREA);
        cv::imshow(WINDOW_NAME, out);
        char c = (char)cv::waitKey(10);
        if( c == 27 )
        {
            break;
        }
    }
    capture.release();
    cv::destroyAllWindows();

    std::cout << "[INFO] Skin segmentation finished on stream." << std::endl;
    return 0;
}