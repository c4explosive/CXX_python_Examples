#define asCxxProg 0

#include <iostream>
#include "EuclideanDist.hpp"

#if not asCxxProg
#include <Python.h>

static PyObject* euclidean_dist(PyObject* self,
                                PyObject* args)
{
    int x0, y0, x1, y1;

    if(!PyArg_ParseTuple(args, "iiii", &x0, &y0, &x1, &y1))
        return NULL;

    EuclideanDist edst(x0, y0, x1, y1);

    return Py_BuildValue("d", edst());

}

static PyMethodDef myMethods[] = {
    {"euclidean_dist", euclidean_dist, METH_VARARGS, "Calculate euclidean dist."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cxxOop =
{
    PyModuleDef_HEAD_INIT,
    "cxxOop",
    "Object oriented in cxx computing euclidean distance.",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_cxxOop(void)
{
    return PyModule_Create(&cxxOop);
}

#else
int main(int argc, char* argv[])
{
    EuclideanDist edst(8,2,7,4);
    std::cout << "Dist:: " << edst() << std::endl;
    return 0;
}
#endif