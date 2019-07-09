#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class Histogram
{
    private:
    cv::Mat imgb, imgg, imgr;
    int binSize[1];
    float hranges[2];
    const float* ranges[1];
    int channels[1];
    int nimages = 1;

    public:
    Histogram(cv::Mat img);
    void computeHistogram(cv::Mat& histb, cv::Mat& histg, cv::Mat& histr); //hist is out
    void operator()(cv::Mat& histb, cv::Mat& histg, cv::Mat& histr);
};

#endif