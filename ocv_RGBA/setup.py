from distutils.core import setup, Extension
import os
import shutil
import glob

setup(name= "cxxRGBA", version = "1.0",
      ext_modules = [
          Extension (
              "cxxRGBA",
              sources = ["cxx_rgba.cxx", "AlphaBlender.cxx"],
              language="c++",
              include_dirs=["/usr/include/opencv4", "/usr/include", 
              "/usr/lib/python3.7/site-packages/numpy/core/include"],
              library_dirs=["/usr/lib64"],
              libraries=["opencv_core", "opencv_highgui", "opencv_imgcodecs",
              "opencv_imgproc", "opencv_dnn", "OpenCL"]
          )
      ]
)

zpath = "build/lib.linux-x86_64-3.7"

files = glob.glob(os.path.join(zpath,"*.so"))

print(files)

shutil.copy(files[0], "./cxxRGBA.so")
