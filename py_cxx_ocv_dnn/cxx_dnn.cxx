#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <numpy/arrayobject.h>
#include <Python.h>

#include "Detector.hpp"

static PyObject* dnn_bridge(PyObject* self,
                            PyObject* args)
{
    PyObject *pimg;

    if(!PyArg_ParseTuple(args, "O", &pimg))
        return NULL;
    
    import_array();

    auto imdims = PyArray_DIMS(pimg);

    cv::Mat img(imdims[0], imdims[1], CV_8UC3, 
                PyArray_DATA(pimg));

    std::vector<cv::Vec4i> coords;
    std::vector<cv::Vec2f> typeScores;
    cv::Mat mdetections;
    Detector dct(img);
    dct(mdetections);
    
    //std::cout << "N:: " << mdetections << std::endl;

    npy_intp mdims [] = {mdetections.rows, mdetections.cols};

    PyObject * tensorRes = PyArray_SimpleNew(2, mdims, NPY_FLOAT32);
    memcpy(PyArray_DATA(tensorRes), mdetections.data, mdims[0]*mdims[1]*sizeof(float));

    return Py_BuildValue("O", tensorRes);
}

static PyMethodDef myMethods[] = {
    {"dnn_bridge", dnn_bridge, METH_VARARGS, "Deep learning SSD/VGG."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cxxDnn =
{
    PyModuleDef_HEAD_INIT,
    "cxxDnn",
    "Deep neural network module, powered in c++.",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_cxxDnn(void)
{
    return PyModule_Create(&cxxDnn);
}