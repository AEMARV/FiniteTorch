import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='mdconv',
      ext_modules=[CUDAExtension('mdconv', ['mdconv.cpp','mdconv_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})