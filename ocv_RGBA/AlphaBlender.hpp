#ifndef ALPHA_BLENDER_HPP
#define ALPHA_BLENDER_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <CL/cl.hpp>

class AlphaBlender
{
private:
    cv::Mat img, plane, imgMask;
    bool loadedCl;

    //cl Context
    cl::Program prog; //load a program for use later
    cl::Context context;
    cl::Device def_dev;
    //cl

    void cvt32C3();
    void ocl_runtime(float * A, float* B, 
                        float* C, float *D, int* N); // C is Out
    void ocl_initContext();
    std::string getclKernelString();
public:
    AlphaBlender(const cv::Mat& img, const cv::Mat& plane, 
                    const cv::Mat imgMask);
    AlphaBlender(); 
    void loadClKernel();
    void blend(cv::Mat& imgO);
    void power_blend(cv::Mat& imgO);
    void ocl_blend(cv::Mat& imgO);
    void changeData(const cv::Mat& img, const cv::Mat& plane, 
                    const cv::Mat& imgMask);

    void operator()(cv::Mat& imgO);
    void operator()(cv::Mat& imgO, bool power);
    void operator()(cv::Mat& imgO, bool power, bool power2);
    ~AlphaBlender();
};


#endif