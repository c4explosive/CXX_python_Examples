#include "Histogram.hpp"
#include <iostream>

Histogram::Histogram(cv::Mat img)
{
    std::vector<cv::Mat> bgr;
    //cv::cvtColor(img, img, cv::COLOR_BGR2Lab);
    cv::split(img, bgr);
    this->imgb = bgr[0].clone();
    this->imgg = bgr[1].clone();
    this->imgr = bgr[2].clone();
    binSize[0] = 256;
    hranges[0] = 0.0;
    hranges[1] = 256.0;
    ranges[0] = hranges;
    channels[0] = 0;
}

void Histogram::computeHistogram(cv::Mat& histb, cv::Mat& histg, cv::Mat& histr)
{
    cv::calcHist(&imgb, 1, channels, cv::Mat(), 
                histb, 1, binSize, ranges);
    cv::calcHist(&imgg, 1, channels, cv::Mat(), 
                histg, 1, binSize, ranges);
    cv::calcHist(&imgr, 1, channels, cv::Mat(), 
                histr, 1, binSize, ranges);
    
}

void Histogram::operator()(cv::Mat& histb, cv::Mat& histg, cv::Mat& histr)
{
    this->computeHistogram(histb, histg, histr);
}