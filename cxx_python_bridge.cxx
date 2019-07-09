#include <Python.h>

#include <iostream>

static PyObject* helloworld(PyObject* self, 
                            PyObject* args)
{
    std::cout << "Hello World, PyCxts_iter3..." << std::endl;
    return Py_None;
}

static PyMethodDef myMethods[] = {
    {"helloworld", helloworld, METH_NOARGS, "Prints Hello world"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "myModule",
    "Test Module",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_myModule(void)
{
    return PyModule_Create(&myModule);
}