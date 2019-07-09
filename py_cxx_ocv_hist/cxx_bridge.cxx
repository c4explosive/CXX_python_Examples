#define mainProg 0
#include <iostream>

//Work with numpy bridge, and with numpy structure put the histogram

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Histogram.hpp"

#if not mainProg
#include <numpy/arrayobject.h>

#include <Python.h>

void packHists(cv::Mat &hist, PyObject *&his, double &max)
{
    double min;
    npy_intp hdim[] = {hist.rows, hist.cols};
    hist.convertTo(hist, CV_32S);
    cv::minMaxLoc(hist, &min, &max, 0, 0);

    his = PyArray_SimpleNew(1, hdim, NPY_INT);
    memcpy(PyArray_DATA(his), hist.data, hdim[0]*sizeof(int));
}

static PyObject* numpy_bridge(PyObject* self,
                              PyObject* args)
{
    PyObject *pimg;
    if(!PyArg_ParseTuple(args, "O", &pimg))
        return NULL;
    
    import_array() // Like: import numpy, is essential!!
    //char * imdata = static_cast<char *>(PyArray_DATA(pimg)); // The numpy is now a pointer, easy as this!!
    auto imdims = PyArray_DIMS(pimg);
    //cv::Mat img(imdims[0], imdims[1], CV_8UC3, imdata);
    cv::Mat img(imdims[0], imdims[1], CV_8UC3);

    cv::Mat mat33(3, 3, CV_8UC3, cv::Scalar(0,0,0));
    int coef = 3;
    for(int i=0; i<mat33.rows; i++)
        for(int j=0; j<mat33.cols; j++)
        {
            mat33.at<cv::Vec3b>(i,j) = cv::Vec3b(coef*1, coef*2, coef*3);
            coef++;
        }


    //std::cout << "Mat33:: " << cv::format(mat33, cv::Formatter::FMT_NUMPY) << std::endl;

    memcpy(img.data, PyArray_DATA(pimg), imdims[0]*imdims[1]*3*sizeof(uchar));

    cv::Mat histb, histg, histr;
    Histogram mHist(img);
    mHist(histb, histg, histr);


    PyObject *hisb, *hisg, *hisr;
    double min, bmax, gmax, rmax;

    packHists(histb, hisb, bmax);
    packHists(histg, hisg, gmax);
    packHists(histr, hisr, rmax);

    // Multidimensional from OpenCV
    npy_intp mat33dim[] = {mat33.rows, mat33.cols, 3};
    PyObject* pMat33 = PyArray_SimpleNew(3, mat33dim, NPY_UINT8);
    memcpy(PyArray_DATA(pMat33), mat33.data, mat33dim[0]*mat33dim[1]*mat33dim[2]*sizeof(uchar));

    auto ohisb = Py_BuildValue("[OOO][ddd]O", hisb, hisg, hisr, bmax, gmax, rmax, pMat33);

    return ohisb;
}

static PyMethodDef myMethods[] = {
    {"numpy_bridge", numpy_bridge, METH_VARARGS, "Numpy data..."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cxxBridge =
{
    PyModuleDef_HEAD_INIT,
    "cxxBridge",
    "Numpy and histogram bridge.",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_cxxBridge(void)
{
    return PyModule_Create(&cxxBridge);
}
#else
int main(int argc, char* argv[])
{
    return 0;
}
#endif