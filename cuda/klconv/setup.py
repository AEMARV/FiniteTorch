from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='klconvs',
      ext_modules=[CUDAExtension('klconvs', ['klconvs.cpp','klconvs_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})