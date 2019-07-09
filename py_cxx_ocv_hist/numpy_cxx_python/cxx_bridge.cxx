#define mainProg 0
#include <iostream>

//Work with numpy bridge, and with numpy structure put the histogram

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <numpy/arrayobject.h>

#if not mainProg

#include <Python.h>

static PyObject* numpy_bridge(PyObject* self,
                              PyObject* args)
{
    std::cout << "Numpy will heres..." << std::endl;
    PyObject * pob, *pobma;
    if(!PyArg_ParseTuple(args, "OO", &pob, &pobma))
        return NULL;
    
    import_array() // Like: import numpy
    char * data = static_cast<char *>(PyArray_DATA(pob)); // The numpy is now a pointer, easy as this!!
    auto dims = PyArray_DIMS(pob);

    char * mdata = static_cast<char *>(PyArray_DATA(pobma));
    auto mdims = PyArray_DIMS(pobma);

    cv::Mat vect(dims[0], dims[1], CV_8U, data);
    cv::Mat mvect(mdims[0], mdims[1], CV_8UC3, mdata);

    vect+=5;
    mvect+=cv::Scalar(1, 1, 50);

    std::cout << "Pob:: " << (void *) pob << std::endl;
    std::cout << "data:: " << (int) data[3] << std::endl;
    std::cout << "Vect:: " << vect << std::endl;
    //std::cout << "Multi:: " << cv::format(mvect, cv::Formatter::FMT_NUMPY) << std::endl;
    cv::namedWindow("+s+CXX", cv::WINDOW_NORMAL);
    cv::imshow("+s+CXX", mvect);
    cv::waitKey(0);
    auto nn = PyArray_SimpleNewFromData(mdims[2], mdims, NPY_UINT8, mdata);
    return nn;
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