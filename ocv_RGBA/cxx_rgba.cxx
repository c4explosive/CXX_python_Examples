#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <numpy/arrayobject.h>
#include <Python.h>

#include "AlphaBlender.hpp"


static PyObject* alpha_process(PyObject* self,
                               PyObject* args)
{
    PyObject *pimgM, *pimg, *pimgB;

    if(!PyArg_ParseTuple(args, "OOO", &pimgM, &pimg, &pimgB))
        return NULL;
    
    //std::cout << "pimgB:: " << (pimgB == Py_None) << std::endl;

    import_array();

    npy_intp* mdim = PyArray_DIMS(pimgM);
    npy_intp* idim = PyArray_DIMS(pimg);
    cv::Mat plane;

    if(pimgB != Py_None)
    {
        npy_intp* bdim = PyArray_DIMS(pimgB);
        plane = cv::Mat(bdim[0], bdim[1], CV_8UC3, PyArray_DATA(pimgB));
    }
    else
        plane = cv::Mat(idim[0], idim[1], CV_8UC3, cv::Scalar(0,0,255));

    cv::Mat imgMask(mdim[0], mdim[1], CV_8U, PyArray_DATA(pimgM));
    cv::cvtColor(imgMask, imgMask, cv::COLOR_GRAY2BGR);
    cv::Mat img(idim[0], idim[1], CV_8UC3, PyArray_DATA(pimg));

    cv::Mat imgO = img.clone();

    AlphaBlender abl(img, plane, imgMask);

    abl(img, true, true); // OCL -> true true, PTR -> true, OCV -> ()

    #define INTERNAL_DEBUG 0


    #if INTERNAL_DEBUG
    cv::namedWindow("+s_org", cv::WINDOW_NORMAL);
    cv::imshow("+s_org", imgO);
    cv::namedWindow("+s_cxx", cv::WINDOW_NORMAL);
    cv::imshow("+s_cxx", img);
    cv::waitKey(0);
    //std::cout << "img_32f:: " << img << std::endl;
    return Py_None;
    #else
    memcpy(PyArray_DATA(pimg), img.data, 
                    3*idim[0]*idim[0]*sizeof(uchar));
    std::cout << "Success!" << std::endl;
    return Py_BuildValue("O", pimg);
    #endif
}

static PyMethodDef myMethods[] = {
    {"alpha_process", alpha_process, METH_VARARGS, "Alpha blending mask..."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cxxRGBA =
{
    PyModuleDef_HEAD_INIT,
    "cxxRGBA",
    "Alpha experiments with RGB and alpha channel",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_cxxRGBA(void)
{
    return PyModule_Create(&cxxRGBA);
}