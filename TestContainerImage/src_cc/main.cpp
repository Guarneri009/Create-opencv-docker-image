
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

int main()
{
    std::cout << cv::getBuildInformation() << std::endl;
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    // CPU
    cv::Mat imgSrc = cv::imread("./resource/lena.jpg");
    cv::imshow("CPU", imgSrc);

    // GPU
    cv::Mat imgDst;
    cv::cuda::GpuMat imgGpuSrc, imgGpuDst;
    imgGpuSrc.upload(imgSrc);
    imgGpuSrc.download(imgDst);
    cv::imshow("GPU", imgDst);

    cv::waitKey(5000);
    return 0;
}
