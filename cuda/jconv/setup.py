import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='jconv',
      ext_modules=[CUDAExtension('jconv', ['jconv.cpp','jconv_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})