#include <torch/torch.h>
#include <iostream>
#include <vector>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand.h>


// Cuda Declarations
at::Tensor klconvs_cuda_forward(at::Tensor input,at::Tensor log_filt,at::Tensor pad);
std::vector<at::Tensor> klconvs_cuda_backward(at::Tensor grad_out, at::Tensor input,at::Tensor log_filt,at::Tensor pad);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// forward declaration cpp
at::Tensor klconvs_forward(at::Tensor input, at::Tensor log_filt, at::Tensor pad
){

  CHECK_INPUT(input);
  CHECK_INPUT(log_filt);
  CHECK_INPUT(pad);

return klconvs_cuda_forward(input, log_filt, pad);
}

//backward declaration cpp
std::vector<at::Tensor> klconvs_backward(at::Tensor grad_output,
at::Tensor input,
at::Tensor log_filt,
at::Tensor pad){
  CHECK_INPUT(input);
  CHECK_INPUT(log_filt);
  CHECK_INPUT(pad);
  CHECK_INPUT(grad_output);
  return klconvs_cuda_backward(input,grad_output, log_filt, pad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &klconvs_forward, "klconvs forward");
  m.def("backward", &klconvs_backward, "klconvs backward");
}