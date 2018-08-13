from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='stochgrads',
      ext_modules=[CUDAExtension('stochgrads', ['stochgrads.cpp','mixer_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})