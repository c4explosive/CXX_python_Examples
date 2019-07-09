#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

class Detector
{
private:
    cv::Mat img;
    void forward(cv::Mat &cno);
    void tensorIterator(const cv::Mat &cno,
                        std::vector<cv::Mat>& detections);
public:
    Detector(const cv::Mat& img);
    void detect(cv::Mat& mdetections);
    void operator()(cv::Mat& mdetections);
    ~Detector();
};



#endif