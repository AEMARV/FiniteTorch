#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand_kernel.h>
// Kernels      ***********************

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__device__ __forceinline__ void calcLinInd(int h, int w, int c, int b, size_t HD, size_t WD,size_t CD, int *out ){
*out = w + WD*(h + HD*(c + CD*(b)));
    return ;
}

template <typename scalar_t>
__global__ void klconvs_cuda_forward_kernel(
    unsigned int seed,
    const scalar_t *input,
    const scalar_t *p_filt,
    scalar_t *out,
    const scalar_t *pad,
    const size_t filt_h,
    const size_t filt_w,
    const size_t filt_c,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b
    ){
const int out_w_sz = blockDim.x, out_h_sz = blockDim.y;
const int out_chan_sz = gridDim.x, batchsz = gridDim.y;
const int w = threadIdx.x , h = threadIdx.y;
const int out_chan_id = blockIdx.x, batch_id=  blockIdx.y;
const int out_idx = threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*(blockIdx.x + gridDim.x*(blockIdx.y)));
curandState_t state;
//for( int channel =0 : channel < )
float temp_reg;
float this_px_out = 0;
bool flag = true;
bool isoutboundh = false;
bool isoutbound = false;
float randnum = 0;
int input_idx = 0;
int filt_idx = 0;
for (int dh=0 ; dh< out_h_sz; dh++){
    isoutboundh = dh + h > inp_h;
    for ( int dw = 0 ; dw< out_w_sz; dw++ ){
        isoutbound = isoutboundh | dw + w > inp_w;
        curand_init(seed,out_idx,dw + dh*(out_w_sz),&state);
        randnum = curand_uniform(&state);
        flag = true;
        for  ( int chan = 0 ; chan < out_chan_sz; chan++){
            // find the correct index of filt
            // get the index val from input
            // add to final answer;
            calcLinInd(dh,dw,chan,out_chan_id,filt_h,filt_w,filt_c, &filt_idx);//[out_chan_id][chan][dh][dw]
            temp_reg = p_filt[filt_idx];
            if (temp_reg > randnum && flag) {
                if (!isoutbound){
                calcLinInd(h+dh ,w+dw ,chan,batch_id, inp_h, inp_w, inp_c, &input_idx);
                this_px_out += input[input_idx];
                }
                flag = false;
            }
            else{
                randnum = randnum - temp_reg;
            }

        }


    }


}
out[out_idx] = this_px_out;

/*
curand_init(seed,0,1,&state);
float randnum = curand_uniform(&state);
curand_init(seed+1,0,0,&state);
float randnum2 = curand_uniform(&state);
printf("time : %f , %f --\n",randnum, randnum2);*/

//printf("out_idx %d: left to right %d,%d,%d,%d   with dims %d,%d,%d,%d\n", out_idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
///__shared__ float[32][32] input_tile; // define shared memory
}
template <typename scalar_t>
__global__ void klconvs_cuda_backward_kernel(){}

// End Kernels *************************

//Forward wrapper ----------------------
at::Tensor klconvs_cuda_forward(
at::Tensor input,
at::Tensor log_filt,
at::Tensor pad){

        at::Tensor p_filt = at::exp(log_filt);
        printf("I AM HEREEEEEEE\n");
        const auto batch_sz = input.size(0);
        const auto im_height = input.size(2);
        const auto im_width = input.size(3);
        const auto im_nchans = input.size(1);
        //CURAND TEST
        //

        //
        // End CURand Test
        printf("batchsz:%d ",batch_sz);

        const auto filt_num = p_filt.size(0);
        const auto filt_height = p_filt.size(2);
        const auto filt_width = p_filt.size(3);
        printf("filt_num:%d ",filt_num);
        at::Tensor out = at::zeros(input.type(),{batch_sz,filt_num,im_height,im_width});

        const dim3 blocks2d(filt_num,batch_sz);
        const dim3 threadsperblock(im_height,im_width);
        const auto seed = time(NULL);
        AT_DISPATCH_FLOATING_TYPES(input.type(), "klconvs_forward_cuda", ([&] {
           klconvs_cuda_forward_kernel<scalar_t><<<blocks2d, threadsperblock>>>(seed,
              input.data<scalar_t>(),
              p_filt.data<scalar_t>(),
              out.data<scalar_t>(),
              pad.data<scalar_t>(),
              filt_width,
              filt_height,
              im_nchans,
              filt_num,
              im_width,
              im_height,
              im_nchans,
              batch_sz

                );
          }));
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        return out;

}
//----------------------------------------------
// Backward wrapper
std::vector<at::Tensor> klconvs_cuda_backward(at::Tensor grad_out,
at::Tensor input,
at::Tensor log_filt,
at::Tensor pad){
return {input, log_filt};
}
