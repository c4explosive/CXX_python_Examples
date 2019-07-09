#include "AlphaBlender.hpp"

#include <CL/cl.hpp>

#include <fstream>
#include <sstream>

std::string AlphaBlender::getclKernelString()
{
    return R"( 
    void kernel alpha_blending(global const float* A, global const float* B, global float* C, global float* D,
                                  global const int* N) 
    {
        int ID, Nthreads, n, ratio, start, stop;

        ID = get_global_id(0);
        Nthreads = get_global_size(0);
        n = N[0];

        ratio = (n / Nthreads);  // number of elements for each thread
        start = ratio * ID;
        stop  = ratio * (ID + 1);

        for (int i=start; i<stop; i++) // A -> alpha, B-> bg, D-> fg, C -> out
            *(C+i) =  *(A+i) * *(D+i) + (1-*(A+i)) * *(B+i); // out = alpha*fg + (1-alpha)*bg
    }
    )";
}

AlphaBlender::AlphaBlender(const cv::Mat& img, const cv::Mat& plane, 
                    const cv::Mat imgMask)
{
    assert(img.size == plane.size && img.size == imgMask.size &&
           plane.size == imgMask.size);

    this->img = img.clone();
    this->plane = plane.clone();
    this->imgMask = imgMask.clone();
    this->loadedCl = false;

}

AlphaBlender::AlphaBlender()
{

}

void AlphaBlender::cvt32C3()
{
    img.convertTo(img, CV_32FC3);
    plane.convertTo(plane, CV_32FC3);
    imgMask.convertTo(imgMask, CV_32FC3, 1/255.);
}

void AlphaBlender::ocl_initContext()
{
    std::cout << "[005 | INF: Initialize OpenCL Runtime]" << std::endl;
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    cl::Platform def_plat = all_platforms[0]; // CPU -> 0
    std::vector<cl::Device> all_devs;
    def_plat.getDevices(CL_DEVICE_TYPE_ALL, &all_devs);
    cl::Device def_dev = all_devs[0];
    cl::Context context({def_dev});

    this->context = context;
    this->def_dev = def_dev;
}

#define KERNEL_OUTSIDE 0

void AlphaBlender::loadClKernel()
{
    this->ocl_initContext();
    cl::Context context = this->context;
    cl::Device def_dev = this->def_dev;
    cl::Program::Sources src;

    cl::Program prog;

    #if KERNEL_OUTSIDE
    std::fstream f("kernel_alpha_blend.cl");
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string kernel_m = ss.str();
    #else
    std::string kernel_m = this->getclKernelString();
    #endif

    src.push_back({kernel_m.c_str(), kernel_m.length()});
    prog = cl::Program(context, src);
    prog.build({def_dev}); // This object can save for later

    this->loadedCl = true;
    this->prog = prog;
}

void AlphaBlender::ocl_runtime(float * A, float* B, 
                        float* C, float *D, int* N)
{
    // Math kernel
    if(!this->loadedCl) // OpenCL is only one time
        this->loadClKernel();


    //int N[1] = {10}; //elements
    long n = N[0];

    //std::cout << "N:" << n << std::endl;
    //return;

    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float)*n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(float)*n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(float)*n);
    cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, sizeof(float)*n);
    cl::Buffer buffer_N(context, CL_MEM_READ_ONLY, sizeof(int));


    cl::CommandQueue queue(context, def_dev);

    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float)*n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float)*n, B);
    queue.enqueueWriteBuffer(buffer_D, CL_TRUE, 0, sizeof(float)*n, D);
    queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(int), N);

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
            alpha_blending(cl::Kernel(prog, "alpha_blending"));

    cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(10), 
                    cl::NullRange);
    
    alpha_blending(eargs, buffer_A, buffer_B, buffer_C, buffer_D, buffer_N).wait();


    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float)*n, C);

    #define SEE_DATA 0
    #if SEE_DATA
    std::cout << "Res: {";
    for(int i=0; i<20; i++)
        std::cout << C[i] << " ";
    std::cout << "}" << std::endl;
    #endif
}

void AlphaBlender::ocl_blend(cv::Mat& imgO)
{
    //std::cout << "OpenCL enter in the rail..." << std::endl;
    
    this->cvt32C3(); //if not, C becomes ANGRY!!
    int dataLength = img.cols * img.rows * img.channels();
    float * pimg = reinterpret_cast<float *>(img.data);
    float * pplane = reinterpret_cast<float *>(plane.data);
    float * pimgMask = reinterpret_cast<float *>(imgMask.data);

    int N[1] = {dataLength};

    this->ocl_runtime(pimgMask, pimg, pimg, pplane, N);
    img.convertTo(img, CV_8UC3, 1.);
    imgO = img.clone();
}

void AlphaBlender::power_blend(cv::Mat& imgO)
{
    this->cvt32C3(); //if not, C becomes ANGRY!!
    int dataLength = img.cols * img.rows * img.channels();
    
    float * pimg = reinterpret_cast<float *>(img.data);
    float * pplane = reinterpret_cast<float *>(plane.data);
    float * pimgMask = reinterpret_cast<float *>(imgMask.data);
    float * pimgO = reinterpret_cast<float *>(imgO.data);


    for(int i=0; i<dataLength;i++) 
    {
        *(pimg+i) = (*(pimgMask+i))*(*(pplane+i)) + (1-*(pimgMask+i)) * (*(pimg+i)); // alpha*fg + (1-alpha)*bg
        //std::cout << "ptr: " << (void*) pimg << std::endl;
    }
    std::cout << "Here_power_OCL_style" << std::endl;
    img.convertTo(img, CV_8UC3, 1.);

    imgO = img.clone();
}

void AlphaBlender::changeData(const cv::Mat& img, 
                const cv::Mat& plane, const cv::Mat& imgMask)
{
    this->img = img.clone();
    this->plane = plane.clone();
    this->imgMask = imgMask.clone();
}


void AlphaBlender::blend(cv::Mat& imgO)
{
    this->cvt32C3();
    cv::multiply(imgMask, plane, plane); // alpha*img1
    cv::multiply(cv::Scalar(1.0,1.0,1.0)-imgMask, img, img); // beta*img2 where beta = 1-alpha

    img = plane + img; // Like assembly image, sum the parts

    img.convertTo(img, CV_8UC3, 1.);

    imgO = img.clone();
}



void AlphaBlender::operator()(cv::Mat& imgO)
{
    this->blend(imgO);
}

void AlphaBlender::operator()(cv::Mat& imgO, bool mode)
{
    this->power_blend(imgO);
}

void AlphaBlender::operator()(cv::Mat& imgO, bool power, bool power2)
{
    this->ocl_blend(imgO);
}

AlphaBlender::~AlphaBlender()
{
}