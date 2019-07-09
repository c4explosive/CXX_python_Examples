#define asCxxProg 0

#if not asCxxProg
#include <Python.h>
#endif

#include <iostream>

int _factorial(int n)
{
    if (n == 0)
        return 1;
    else
        return n * _factorial(n-1);
}

#if not asCxxProg
static PyObject* factorial(PyObject* self,
                           PyObject* args)
{
    int n; //Our factorial value

    if(!PyArg_ParseTuple(args, "i", &n))
        return NULL;

    return Py_BuildValue("i", _factorial(n));
}

static PyMethodDef myMetods[] = {
    {"factorial", factorial, METH_VARARGS, "Calc the Factorial"}, //METH_VARARGS is for args from python
    {NULL, NULL, 0, NULL} //the "NULL" for end the array
};

static struct PyModuleDef cxxFactorial =
{
    PyModuleDef_HEAD_INIT,
    "cxxFactorial",
    "Factorial in cxx",
    -1,
    myMetods
};

PyMODINIT_FUNC PyInit_cxxFactorial(void)
{
    return PyModule_Create(&cxxFactorial);
}
#endif

#if asCxxProg
int main(int argc, char* argv[])
{
    std::cout << "Main in CXX::: " << _factorial(9) << std::endl;
    return 0;
}
#endif