#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include </usr/local/cuda-9.0/targets/x86_64-linux/include/curand_kernel.h>

// Helper functions
__device__ __forceinline__ size_t calcLinInd4(int idx_a, int idx_b, int idx_c, int idx_d, size_t AD, size_t BD,size_t CD, size_t DD ){
/*    if (idx_d>= DD || idx_d < 0){
    printf("fault1");
    }
    if (idx_c>= CD || idx_c < 0){
    printf("fault2");
    }
    if (idx_c>= BD || idx_b < 0){
    printf("fault3");
    }
    if (idx_c>= AD || idx_a < 0){
    printf("fault4");
    }*/
    return idx_d + DD*(idx_c + CD*(idx_b + BD*(idx_a)));
}
// ********************************************************************************************************************
// ********************************************************************************************************************
// ********************************************************************************************************************
// ********************************************************************************************************************
// ************************************************ Kernels ***********************************************************
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename scalar_t>
__global__ void mdconv_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ l_filt,
    scalar_t* __restrict__ out,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_w,
    const size_t inp_h,
    const size_t inp_c,
    const size_t inp_b,
    const int totalOutPx,
    const int totalTreads,
    const float minusplus
    )
{
    const size_t threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadlinidx < totalOutPx){

        // Calculate Imout Indices
        int int_temp_reg;
        const int im_w_idx = threadlinidx % inp_w;
        int_temp_reg =inp_w;
        const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
        int_temp_reg *=inp_h;
        const int im_c_idx = (threadlinidx/(int_temp_reg)) % filt_n;
        int_temp_reg *=filt_n;
        const int im_b_idx = (threadlinidx/(int_temp_reg));
        size_t size_temp_reg;

        float this_px_out=0;
        float float_temp;
        float this_px_this_pxfilt_result;
        size_t input_idx=0;
        int dh;
        int dw;
        int chan;
        int cur_im_h =0;
        int cur_im_w =0;
        for ( dh= 0 ; dh < filt_h; dh++){
            cur_im_h= dh + im_h_idx - ((filt_h)/2);
            if (cur_im_h< 0){
                    continue;
                            }
            if (cur_im_h >= inp_h){
                    break;
                }

            for (dw = 0 ; dw < filt_w; dw++ ){
                cur_im_w = dw + im_w_idx - ((filt_w)/2);
                if (cur_im_w<0){
                    continue;
                }
                if (cur_im_w >= inp_w){
                    break;
                }
                for  ( chan = 0 ; chan < inp_c; chan++){
                    // find the correct index of filt
                    // get the index val from input
                    // add to final answer;
                    //size_temp_reg = calcLinInd( dh, dw, chan,im_c_idx, filt_h, filt_w, inp_c);
                    //input_idx = calcLinInd(cur_im_h, cur_im_w, chan, im_b_idx, inp_h, inp_w, inp_c);
                    size_temp_reg = calcLinInd4( im_c_idx, chan, dh,dw,filt_n, inp_c, filt_h, filt_w);
                    input_idx = calcLinInd4(  im_b_idx, chan,cur_im_h, cur_im_w,inp_b, inp_c , inp_h, inp_w);
                    float_temp = minusplus*(l_filt[size_temp_reg] - ( input[input_idx]));
                    if (chan==0){
                        this_px_this_pxfilt_result = float_temp;
                    }
                    else{
                        this_px_this_pxfilt_result = fminf(float_temp,this_px_this_pxfilt_result);

                    }
                }
                this_px_out += this_px_this_pxfilt_result;
            }
        }

        out[threadlinidx] = this_px_out; //////////////////// GL MEM WRITE

    }
}

template <typename scalar_t>
__global__ void mdconv_cuda_backward_kernel(
    const scalar_t* __restrict__ input, //TODO: MAKE sure the dims are dzdin and the threads are compatible
    const scalar_t* __restrict__ lfilt,
    const scalar_t* __restrict__ dzdout,
    float* __restrict__ dzdin,
    float* __restrict__ dzdl_filt,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_w,
    const size_t inp_h,
    const size_t inp_c,
    const size_t inp_b,
    const int totalThreads,
    const float minusplus
)
{
   const size_t threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadlinidx < totalThreads){

        // Calculate Imout Indices
        int int_temp_reg;
        const int im_w_idx = threadlinidx % inp_w;
        int_temp_reg =inp_w;
        const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
        int_temp_reg *=inp_h;
        const int im_c_idx = (threadlinidx/(int_temp_reg)) % filt_n;
        int_temp_reg *=filt_n;
        const int im_b_idx = (threadlinidx/(int_temp_reg));
        size_t size_temp_reg;

        float float_temp;
        float min_val_temp;
        int chan_temp;
        size_t input_idx=0;
        int dh;
        int dw;
        int chan;
        int cur_im_h =0;
        int cur_im_w =0;
        for ( dh= 0 ; dh < filt_h; dh++){
            cur_im_h= dh + im_h_idx - ((filt_h)/2);
            if (cur_im_h< 0){
                    continue;
                            }
            if (cur_im_h >= inp_h){
                    break;
                }

            for (dw = 0 ; dw < filt_w; dw++ ){
                cur_im_w = dw + im_w_idx - ((filt_w)/2);
                if (cur_im_w<0){
                    continue;
                }
                if (cur_im_w >= inp_w){
                    break;
                }
                min_val_temp= 0;
                chan_temp = 0;
                for  ( chan = 0 ; chan < inp_c; chan++){
                    // find the correct index of filt
                    // get the index val from input
                    // add to final answer;
                    //size_temp_reg = calcLinInd( dh, dw, chan,im_c_idx, filt_h, filt_w, inp_c);
                    //input_idx = calcLinInd(cur_im_h, cur_im_w, chan, im_b_idx, inp_h, inp_w, inp_c);
                    size_temp_reg = calcLinInd4( im_c_idx, chan, dh,dw,filt_n, inp_c, filt_h, filt_w);
                    input_idx = calcLinInd4(  im_b_idx, chan,cur_im_h, cur_im_w,inp_b, inp_c , inp_h, inp_w);
                    float_temp = minusplus *(lfilt[size_temp_reg] - input[input_idx]); // Memory Access
                    if (float_temp < min_val_temp){
                        min_val_temp = float_temp;
                        chan_temp= chan;//inp_c ;
                    }

                }

                // find the correct index of filt
                // get the index val from input
                // add to final answer;
                //size_temp_reg = calcLinInd( dh, dw, chan,im_c_idx, filt_h, filt_w, inp_c);
                //input_idx = calcLinInd(cur_im_h, cur_im_w, chan, im_b_idx, inp_h, inp_w, inp_c);
                size_temp_reg = calcLinInd4( im_c_idx, chan_temp, dh,dw,filt_n, inp_c, filt_h, filt_w);
                input_idx = calcLinInd4(  im_b_idx, chan_temp,cur_im_h, cur_im_w,inp_b, inp_c , inp_h, inp_w);
                float_temp = minusplus*dzdout[threadlinidx] ; // Memory Access
                atomicAdd(&(dzdin[input_idx]),-float_temp); //Memory Access
                atomicAdd(&(dzdl_filt[size_temp_reg]),float_temp); // Memory Access

            }
        }

    }

}


// ********************************************************************************************************************
// ********************************************************************************************************************
// ********************************************************************************************************************
// ********************************************************************************************************************
// ********************************************************************************************************************
// ------------------------------------Kernel Call Wrappers -----------------------------------------------------------
at::Tensor mdconv_cuda_forward(
at::Tensor input,
at::Tensor log_filt,
const float minusplus
    )
    {

        //at::Tensor p_filt = (at::exp(log_filt));
       // p_filt = p_filt.cumsum(1);
        const auto batch_sz = input.size(0);
        const auto im_height = input.size(2);
        const auto im_width = input.size(3);
        const auto im_nchans = input.size(1);

        const auto filt_num = log_filt.size(0);
        const auto filt_height = log_filt.size(2);
        const auto filt_width = log_filt.size(3);
        //printf("(%d,%d,%d,%d)\n", p_filt.size(0),p_filt.size(1),p_filt.size(2),p_filt.size(3));
        //printf("filt_num:%d ",filt_num);
        auto out = at::zeros(input.type(),{batch_sz,filt_num,im_height,im_width}); //TODO: Remove except zero loop

        auto random = at::rand(input.type(),{filt_height,filt_width,batch_sz,filt_num,im_height,im_width});
        const int totalOutPx = im_height*im_width*batch_sz*filt_num;
        // Single Loop const auto totalThreads = totalOutPx*filt_height*filt_width;
        const  int totalThreads = totalOutPx;
        int j = 25;
        const  int threadsperblock =j*32;
        int blockNum = (totalThreads/threadsperblock);
        if (totalThreads%threadsperblock != 0 ){
            blockNum++;

        }
        const dim3 blocks(blockNum);
        //printf("blocks: %d, totaltherads/threadperbloc : %d", blocks,totalThreads/threadsperblock);

        AT_DISPATCH_FLOATING_TYPES(input.type(), "klconvs_forward_cuda", ([&] {
           mdconv_cuda_forward_kernel<scalar_t><<<blocks, threadsperblock>>>(
              input.data<scalar_t>(),
              log_filt.data<float>(),
              out.data<scalar_t>(),
              filt_width,
              filt_height,
              filt_num,
              im_width,
              im_height,
              im_nchans,
              batch_sz,
              totalOutPx,
              totalThreads,
              minusplus

                );
          }));


        //out = out.sum(0);  /// ZEro Loop Version \TODO: rremove in case of diff kernel
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        return out;
    }


std::vector<at::Tensor> mdconv_cuda_backward(at::Tensor dzdout,
at::Tensor input,
at::Tensor log_filt,
const float minusplus
)
    {
    const auto batch_sz = input.size(0);
    const auto im_height = input.size(2);
    const auto im_width = input.size(3);
    const auto im_nchans = input.size(1);

    const auto filt_num = log_filt.size(0);
    const auto filt_height = log_filt.size(2);
    const auto filt_width = log_filt.size(3);

    auto dzdinput = at::zeros_like(input);
    auto dzdlfilt = at::zeros_like(log_filt);


    const  int totalThreads = im_height*im_width*batch_sz*filt_num;
    int j = 25; //TODO: Make J chosen automatically. the shared memory is the bottleneck.

    const  int threadsperblock =j*32;
    //printf("shared mem bytes %d - KB: %d, j:%d ",shared_per_block, shared_per_block/1024,j);
    int blockNum = (totalThreads/threadsperblock);
    if (totalThreads%threadsperblock != 0 ){
        blockNum++;
    }
    const dim3 blocks(blockNum);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mdconv_backward_cuda", ([&] {
       mdconv_cuda_backward_kernel<scalar_t><<<blocks, threadsperblock>>>(  // TODO : CHANGE KLCONVS AND KLCONV BACK AND FORTH. NO FORGET.... NEVER FORGET, it is easy not to seee.
          input.data<scalar_t>(),
          log_filt.data<scalar_t>(),
          dzdout.data<scalar_t>(),
          dzdinput.data<float>(),
          dzdlfilt.data<float>(),
          filt_width,
          filt_height,
          filt_num,
          im_width,
          im_height,
          im_nchans,
          batch_sz,
          totalThreads,
          minusplus
            );
      }));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    return {dzdinput, dzdlfilt};
    }