#include <torch/torch.h>
#include <iostream>
#include <vector>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand.h>


// Cuda Declarations
std::vector<at::Tensor> klconvs_cuda_forward(at::Tensor input,at::Tensor log_filt);
std::vector<at::Tensor> klconvs_cuda_backward(at::Tensor grad_out, at::Tensor input,at::Tensor log_filt, at::Tensor random);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// forward declaration cpp
std::vector<at::Tensor> klconvs_forward(at::Tensor input, at::Tensor log_filt
){

  CHECK_INPUT(input);
  CHECK_INPUT(log_filt);

return klconvs_cuda_forward(input, log_filt);
}

//backward declaration cpp
std::vector<at::Tensor> klconvs_backward(at::Tensor grad_output,
at::Tensor input,
at::Tensor log_filt,
at::Tensor random){
  CHECK_INPUT(input);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(random);
    CHECK_INPUT(log_filt);
  return klconvs_cuda_backward(grad_output, input, log_filt, random);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &klconvs_forward, "klconvs forward");
  m.def("backward", &klconvs_backward, "klconvs backward");
}