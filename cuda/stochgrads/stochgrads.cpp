
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand.h>


std::vector<at::Tensor> mixer_cuda_backward(at::Tensor dzdout, at::Tensor input, at::Tensor output, at::Tensor log_filt);
//std::vector<at::Tensor> logsoft_cuda_backward(at::Tensor grad_out, int dim);


#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> mixer_backward(at::Tensor dzdout , at::Tensor input, at::Tensor output, at::Tensor log_filt
){
  CHECK_INPUT(dzdout);
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(log_filt);

return mixer_cuda_backward(dzdout , input, output,  log_filt);
}
/*
std::vector<at::Tensor> logsoft_backward(at::Tensor dzdout, at::Tensor input, int dim
){
  CHECK_INPUT(dzdout);
  CHECK_INPUT(input);

return logsoft_cuda_backward(dzdout, int dim);
}*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mixer_backward", &mixer_backward, "mixer backward");
  //m.def("logsoft_backward", &logsoft_backward, "Log Softmax Backward");
}