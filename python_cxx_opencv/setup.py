from distutils.core import setup, Extension
import os

#os.environ["CC"] = "g++"

setup(name = "cxxOpencv", version = "1.0",
       ext_modules = [ 
           Extension("cxxOpencv", 
           ["cxx_opencv.cxx"],
           language="c++",
           include_dirs=["/usr/include/opencv4", "/usr/include/"], # The headers are here (1st compiling command)
           library_dirs=["/usr/lib64"], # The real libraries are here (2nd compilig command)
           libraries=["opencv_core", "opencv_highgui", "opencv_imgcodecs"] # The modules of opencv are here, if one missing, a beutiful RUNTIME's linked error will appear!!!
       )])
