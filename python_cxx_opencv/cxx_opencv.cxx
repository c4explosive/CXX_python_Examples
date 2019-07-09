#define asCxxProg 0

#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#if not asCxxProg
#include <Python.h>

static PyObject* mat_see_data(PyObject* self,
                              PyObject* args)
{
    char* filename;

    if(!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    //std::cout << "Name::: " << std::string(filename) << std::endl;
    cv::Mat mat(1000, 500, CV_8UC3, cv::Scalar(255, 0, 0));
    mat = cv::imread(std::string(filename));
    //std::cout << "Mat_zz: " << mat.size << std::endl;
    return Py_BuildValue("ii", mat.size[0], mat.size[1]);
}

static PyMethodDef myMethods[] = {
    {"mat_see_data", mat_see_data, METH_VARARGS, 
     "See a real cv::Mat dims."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cxxOpencv =
{
    PyModuleDef_HEAD_INIT,
    "cxxOpencv",
    "Real opencv in CXX.",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_cxxOpencv(void)
{
    return PyModule_Create(&cxxOpencv);
}

#else
int main(int argc, char* argv[])
{
    cv::Mat mat(1000, 500, CV_8UC3, cv::Scalar(255, 0, 0));
    mat = cv::imread("file.jpg");
    std::cout << "Mat: " << mat.size << std::endl;
    return 0;
}
#endif