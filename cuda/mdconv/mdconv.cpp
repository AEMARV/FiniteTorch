#include <torch/torch.h>
#include <iostream>
#include <vector>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand.h>


// Cuda Declarations
at::Tensor mdconv_cuda_forward(at::Tensor log_input,at::Tensor log_filt, const float plusminus);
std::vector<at::Tensor> mdconv_cuda_backward(at::Tensor grad_out, at::Tensor log_input,at::Tensor log_filt, const float plusminus);

// Macros
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor mdconv_forward(at::Tensor input, at::Tensor log_filt
){

  CHECK_INPUT(input);
  CHECK_INPUT(log_filt);
  const float plusminus = 1;
return mdconv_cuda_forward(input, log_filt, plusminus);
}

at::Tensor mdconv_i_forward(at::Tensor input, at::Tensor log_filt
){

  CHECK_INPUT(input);
  CHECK_INPUT(log_filt);
  const float plusminus = -1;
return mdconv_cuda_forward(input, log_filt, plusminus);
}

//backward declaration cpp
std::vector<at::Tensor> mdconv_backward(at::Tensor grad_output,
at::Tensor input,
at::Tensor log_filt){
  CHECK_INPUT(input);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(log_filt);
  const float plusminus = 1;
  return mdconv_cuda_backward(grad_output, input, log_filt,plusminus);
}

std::vector<at::Tensor> mdconv_i_backward(at::Tensor grad_output,
at::Tensor input,
at::Tensor log_filt){
  CHECK_INPUT(input);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(log_filt);
  const float plusminus = -1;
  return mdconv_cuda_backward(grad_output, input, log_filt,plusminus);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mdconv_forward, "mdconv forward");
  m.def("backward", &mdconv_backward, "mdconv backward");
  m.def("iforward", &mdconv_i_forward, "mdconv inverse forward");
  m.def("ibackward", &mdconv_i_backward, "mdconv inverse backward");
}