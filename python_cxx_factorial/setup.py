from distutils.core import setup, Extension

setup(name = "cxxFactorial", version = "1.0",
        ext_modules = [Extension("cxxFactorial",
        ["cxx_factorial.cxx"])])