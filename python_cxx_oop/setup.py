from distutils.core import setup, Extension

setup(name = "cxxOop", version = "1.0",
        ext_modules = [ Extension("cxxOop",
        ["cxx_oop.cxx"])])