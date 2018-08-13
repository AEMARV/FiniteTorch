#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand_kernel.h>

// *********************************************HELPER FUNCS************************************************************
// Error Check Macros

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__ __forceinline__ size_t calcLinInd4(int idx_a, int idx_b, int idx_c, int idx_d, size_t AD, size_t BD,size_t CD, size_t DD ){
// Calculates the 4 dimensional linear indices.
// The indices from left to right is outermost to innermost, the size of of the the outer most is not used.

    return idx_d + DD*(idx_c + CD*(idx_b + BD*(idx_a)));
}
// *****************************************END-HELPER FUNCS************************************************************

//////////////////// KERNEL ////////////////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void mixer_cuda_backward_kernel(
    const scalar_t* __restrict__ dzdout,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ output,
    const scalar_t* __restrict__ lfilt,
    const scalar_t* __restrict__ random,
    float* __restrict__ dzdinput,
    float* __restrict__ dzdlfilt,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const size_t totalThreads
    ){
    const size_t threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (threadlinidx < totalThreads){

        // Shared Memory Config
//        int threadid = threadIdx.x;
//        extern __shared__ float s[];
//        float *ScratchPad = (float*)&s[threadid*inp_c];
        // End Shared Memory Config

        int int_temp_reg;
        const int im_w_idx = threadlinidx % inp_w;
        int_temp_reg =inp_w;
        const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
        int_temp_reg *=inp_h;
        const int filt_n_idx = (threadlinidx/(int_temp_reg)) % filt_n;
        int_temp_reg *=filt_n;
        const int im_b_idx = (threadlinidx/(int_temp_reg)) % inp_b;
        size_t filt_idx;
        size_t inp_idx;
        size_t out_idx = calcLinInd4(im_b_idx, filt_n_idx, im_h_idx, im_w_idx , inp_b, filt_n, inp_h, inp_w);
        float temp;
        size_t input_final_idx = 0;
        size_t filt_final_idx = 0;
        float out_this = output[out_idx];   // TODO : GL MEM ******
        float dzdout_this = dzdout[out_idx]; // TODO : GL MEM ******
        float rand = random[threadlinidx]; // TODO: WatchOut randoms are read from a stream

        for (int chan = 0 ; chan< inp_c ; chan++){
            filt_idx = calcLinInd4(filt_n_idx, chan , 0,0, filt_n, inp_c,1,1);
            inp_idx = calcLinInd4(im_b_idx, chan , im_h_idx, im_w_idx, inp_b, inp_c,inp_h,inp_w);
            temp = expf(lfilt[filt_idx] + input[inp_idx] - out_this); // TODO : GL MEM ******
            if (rand < temp){
               input_final_idx = inp_idx;
               filt_final_idx = filt_idx;
               break;
            }
            rand -= temp;
        }

        atomicAdd(&(dzdinput[input_final_idx]), dzdout_this); // TODO : GL MEM ******
        atomicAdd(&(dzdlfilt[filt_final_idx]), dzdout_this); // TODO : GL MEM ******

    }

}

//////////////////// END- KERNEL ///////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> mixer_cuda_backward(at::Tensor dzdout,
at::Tensor input,
at::Tensor output,
at::Tensor log_filt
){

const auto batch_sz = input.size(0);
const auto im_height = input.size(2);
const auto im_width = input.size(3);
const auto im_nchans = input.size(1);

const auto filt_num = log_filt.size(0);
const auto filt_height = 1;
const auto filt_width = 1;

auto dzdinput = at::zeros_like(input);
auto dzdlfilt = at::zeros_like(log_filt);

auto random = at::rand(input.type(),{batch_sz,filt_num,im_height,im_width});
// Single Loop const auto totalThreads = totalOutPx*filt_height*filt_width;
int totalThreads = batch_sz * filt_num * im_height * im_width;
int warps_per_block = 32;
const  int threadsperblock = warps_per_block * 32;
int blockNum = (totalThreads/threadsperblock);
if (totalThreads%threadsperblock != 0 ){
    blockNum++;
}
const dim3 blocks(blockNum);
//printf("blocks: %d, totaltherads/threadperbloc : %d", blocks,totalThreads/threadsperblock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "klconvs_backward_rand_cuda", ([&] {
       mixer_cuda_backward_kernel<scalar_t><<<blocks, threadsperblock>>>(  // TODO : CHANGE KLCONVS AND KLCONV BACK AND FORTH. NO FORGET.... NEVER FORGET, it is easy not to seee.
          dzdout.data<scalar_t>(),
          input.data<scalar_t>(),
          output.data<scalar_t>(),
          log_filt.data<scalar_t>(),
          random.data<scalar_t>(),// rand . data please fix
          dzdinput.data<float>(),
          dzdlfilt.data<float>(),
          filt_width,
          filt_height,
          filt_num,
          im_width,
          im_height,
          im_nchans,
          batch_sz,
          totalThreads
            );
      }));

        //dzdlfilt = at::div(dzdlfilt,im_height*im_width);
        //out = out.sum(0);  /// ZEro Loop Version \TODO: rremove in case of diff kernel
        gpuErrchk( cudaPeekAtLastError() );
        //gpuErrchk( cudaDeviceSynchronize() );



return {dzdinput, dzdlfilt};
}