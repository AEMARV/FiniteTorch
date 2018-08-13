#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
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



__device__ __forceinline__ size_t calcLinInd(int h, int w, int c, int b, size_t HD, size_t WD,size_t CD ){

    return w + WD*(h + HD*(c + CD*(b)));
}
__device__ __forceinline__ void calcIndpInd(int linind, int h, int w, int c, int b, size_t HD, size_t WD,size_t CD,int *out ){
*out = w + WD*(h + HD*(c + CD*(b)));
    return ;
}
__device__ __forceinline__ void calcIndForRandom(int linind, int h, int w, int c, int b, size_t HD, size_t WD,size_t CD,int *out ){
*out = w + WD*(h + HD*(c + CD*(b)));
    return ;
}
__device__ __forceinline__ size_t calcLinInd4(int idx_a, int idx_b, int idx_c, int idx_d, size_t AD, size_t BD,size_t CD, size_t DD ){

    return idx_d + DD*(idx_c + CD*(idx_b + BD*(idx_a)));
}

/*
template <typename scalar_t>
__global__ void klconvs_cuda_forward_kernel_new(
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


curand_init(seed,0,1,&state);
float randnum = curand_uniform(&state);
curand_init(seed+1,0,0,&state);
float randnum2 = curand_uniform(&state);
printf("time : %f , %f --\n",randnum, randnum2);

//printf("out_idx %d: left to right %d,%d,%d,%d   with dims %d,%d,%d,%d\n", out_idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
///__shared__ float[32][32] input_tile; // define shared memory
}
*/
template <typename scalar_t>
__global__ void klconvs_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ l_filt,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ random,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const int totalOutPx,
    const int totalTreads
    ){
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
        float randnum;
        float this_px_out=0;
        float float_temp;
        float log_probs=0;
        size_t input_idx=0;
        int dh;
        int dw;
        int chan;
        int cur_im_h =0;
        int cur_im_w =0;
        int rand_idx = 0;
        float p_filt;
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
                randnum = random[threadlinidx + totalOutPx*(dw +  filt_w*( dh ))];// * isoutbound;  // GLOBAL MEM ACCESS
                for  ( chan = 0 ; chan < inp_c; chan++){
                    // find the correct index of filt
                    // get the index val from input
                    // add to final answer;
                    size_temp_reg = calcLinInd( dh, dw, chan,im_c_idx, filt_h, filt_w, inp_c);
                    float_temp = l_filt[size_temp_reg];
                    p_filt = expf(float_temp);
                    if (randnum <= p_filt){
                        input_idx = calcLinInd(cur_im_h, cur_im_w, chan, im_b_idx, inp_h, inp_w, inp_c);
                        //inp_indices[dh][dw] = input_idx; ////////// GL MEM ACCESS  ////*********** check wether bool*float is float
                        //randnum = 100;
                        log_probs = float_temp;
                        random[threadlinidx + totalOutPx*(dw +  filt_w*( dh ))] = __int2float_rn(chan);
                        break;
                    }
                    //flag = (flag && !flag2);
                    //j = j + flag2;
                    randnum = randnum - p_filt;
                }
                this_px_out += input[input_idx] -log_probs;
            }
        }

        //for (int j=0 ; j< input_idx; j++){
        //    this_px_out += input[inp_indices[j]];
        //}
        out[threadlinidx] = this_px_out; //////////////////// GL MEM WRITE

    }
}

/*
template <typename scalar_t>
__global__ void klconvs_cuda_forward_kernel_single_loop(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ p_filt,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ random,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_c,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const int totalOutPx,
    const int totalThreads
    ){
    const int threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadlinidx < totalThreads){
        // Calculate Imout Indices
        int int_temp_reg =1;
        const int im_w_idx = threadlinidx % inp_w;
        int_temp_reg = inp_w;
        const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
        int_temp_reg = int_temp_reg * inp_h;
        const int im_c_idx = (threadlinidx/(int_temp_reg)) % filt_n;
        int_temp_reg = int_temp_reg * filt_n;
        const int im_b_idx = (threadlinidx/(int_temp_reg)) %inp_b;
        int_temp_reg = int_temp_reg * inp_b;
        const int dw = (threadlinidx/(int_temp_reg)) %filt_w;
        int_temp_reg = int_temp_reg * filt_w;
        const int dh = (threadlinidx/(int_temp_reg)) %filt_h;
        int out_idx = im_w_idx;
        int_temp_reg= inp_w;
        out_idx += im_h_idx * int_temp_reg;
        int_temp_reg *= inp_h;
        out_idx += im_c_idx * int_temp_reg;
        int_temp_reg *= inp_c;
        out_idx += im_b_idx * int_temp_reg;
        float randnum ;
        int rand_idx;
        float temp_reg=0;
        float this_px_out=0;
        // Flags
        bool flag = true;
        bool flag2= false;
        bool isoutboundh = false;
        bool isoutbound = false;

        int input_idx;
        int filt_idx;
        int j = 0;


        isoutboundh = (dh + im_h_idx) > inp_h;
        isoutbound = isoutboundh || ((dw + im_w_idx) > inp_w);
        rand_idx = threadlinidx + totalOutPx*(dw+ filt_w*(dh));
        randnum = random[rand_idx];// * isoutbound;  // GLOBAL MEM ACCESS
        flag = true;
        for  ( int chan = 0 ; chan < filt_c; chan++){
            // find the correct index of filt
            // get the index val from input
            // add to final answer;
            calcLinInd(dh,dw,chan,im_c_idx,filt_h,filt_w,filt_c, &filt_idx);//[out_chan_id][chan][dh][dw]
            //temp_reg = p_filt[filt_idx];  ////////////GLOBAL MEM ACESSS
            flag2 = flag && (temp_reg >= randnum) && (!isoutbound);
            calcLinInd(im_h_idx + dh, im_w_idx + dw, chan, im_b_idx, inp_h, inp_w, inp_c, &input_idx);
            this_px_out = this_px_out + (flag2 * input[input_idx]); ////////// GL MEM ACESSS  ////*********** check wether bool*float is float
            if (flag2){
                break;
            }
            flag = (flag && !flag2);
            j = j + flag2;
            randnum = randnum - temp_reg;
        }

        //atomicAdd(&(out[out_idx]),this_px_out);
       // out[out_idx] += this_px_out; //////////////////// GL MEM WRITE

    }
}

template <typename scalar_t>
__global__ void klconvs_cuda_forward_kernel_zero_loop(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ p_filt,
    scalar_t* out,
    const scalar_t* __restrict__ random,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_c,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const long int totalOutPx,
    const long int totalActiveThreads
    ){
    const long int threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadlinidx < totalActiveThreads){
        // Calculate Imout Indices
        int int_temp_reg =1;
        const int im_w_idx = threadlinidx % inp_w;
        int_temp_reg = inp_w;
        const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
        int_temp_reg = int_temp_reg * inp_h;
        const int im_c_idx = (threadlinidx/(int_temp_reg)) % filt_n;
        int_temp_reg = int_temp_reg * filt_n;
        const int im_b_idx = (threadlinidx/(int_temp_reg)) % inp_b;
        int_temp_reg = int_temp_reg * inp_b;
        const int dw = (threadlinidx/(int_temp_reg)) % filt_w;
        int_temp_reg = int_temp_reg * filt_w;
        const int dh = (threadlinidx/(int_temp_reg)) % filt_h;
        int_temp_reg *= filt_h;
        int chan = (threadlinidx/(int_temp_reg));
        int out_idx = im_w_idx + inp_w*( im_h_idx + inp_h*(im_c_idx + filt_n*(im_b_idx + inp_b*(dw + filt_w*(dh )))));

        float randnum=0.5 ;
        long int rand_idx;
        float current_cumprob_reg=0;
        float prev_cumprob_reg=0;
        // Flags
        bool flag = true;
        bool flag2= false;
        bool isoutboundh = false;
        bool isoutbound = false;

        long int input_idx;
        long int filt_idx;
        long int prev_filt_idx;
        int j = 0;


        isoutboundh = (dh + im_h_idx) >= inp_h;
        isoutbound = isoutboundh || ((dw + im_w_idx) >= inp_w);
        rand_idx = threadlinidx % (inp_w*inp_h*filt_n*inp_b*filt_w*filt_h);
        randnum = random[rand_idx];// * isoutbound;  // GLOBAL MEM ACCESS
        flag = true;

        // find the correct index of filt
        // get the index val from input
        // add to final answer;

        calcLinInd(dh,dw,chan,im_c_idx,filt_h,filt_w,filt_c, &filt_idx);//[out_chan_id][chan][dh][dw]

        current_cumprob_reg = p_filt[filt_idx];  ////////////GLOBAL MEM ACESSS
        if (chan == 0){
            prev_cumprob_reg = 0 ;
        }
        else{
            calcLinInd(dh,dw,(chan-1),im_c_idx,filt_h,filt_w,filt_c, &prev_filt_idx);//[out_chan_id][chan][dh][dw]
            prev_cumprob_reg = p_filt[prev_filt_idx];  ////////////GLOBAL MEM ACESSS
        }
        flag2 = (prev_cumprob_reg < randnum) && (current_cumprob_reg >= randnum) && (!isoutbound);

        //this_px_out =  input[input_idx];
        if (flag2){
            //out_idx = 0;
            calcLinInd(im_h_idx + dh, im_w_idx + dw, chan, im_b_idx, inp_h, inp_w, inp_c, &input_idx);

            //out[out_idx] = atomicAdd(&(out[out_idx]),input[input_idx];////////// GL MEM ACESSS  ////*********** check wether bool*float is float
            if (out_idx > filt_h *filt_w *totalOutPx){
                printf("Culprit: %d", out_idx);
            }
              out[out_idx] += input[input_idx]; //////////////////// GL MEM WRITE

        }

        //atomicAdd(&(out[out_idx]),this_px_out);
       // out[out_idx] += this_px_out; //////////////////// GL MEM WRITE

    }
}
/* ----------------------   Print Tests
printf("number of times went in %d\n",j);
curand_init(seed,0,1,&state);
float randnum = curand_uniform(&state);
curand_init(seed+1,0,0,&state);
float randnum2 = curand_uniform(&state);
printf("time : %f , %f --\n",randnum, randnum2);
printf("out_idx %d: left to right %d,%d,%d,%d   with dims %d,%d,%d,%d\n", out_idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
*/

//__shared__ float[32][32] input_tile; // define shared memory


template <typename scalar_t>
__global__ void klconvs_cuda_backward_kernel(
    const scalar_t* __restrict__ input, //TODO: MAKE sure the dims are dzdin and the threads are compatible
    const scalar_t* __restrict__ lfilt,
    const scalar_t* __restrict__ dzdout,
    const scalar_t* __restrict__ random,
    float* __restrict__ dzdin,
    float* __restrict__ dzdl_filt,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const int totalThreads
){
    const size_t threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    int int_temp_reg;
    const int im_w_idx = threadlinidx % inp_w;
    int_temp_reg =inp_w;
    const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
    int_temp_reg *=inp_h;
    const int filt_n_idx = (threadlinidx/(int_temp_reg)) % filt_n;
    int_temp_reg *=filt_n;
    const int im_b_idx = (threadlinidx/(int_temp_reg)) % inp_b;
    int_temp_reg *= inp_b;
    const int filt_w_idx = (threadlinidx/(int_temp_reg)) % filt_w;
    int_temp_reg *= filt_w;
    const int filt_h_idx = (threadlinidx/(int_temp_reg)) ;

    const int inp_w_idx = im_w_idx + filt_w_idx - (filt_w/2) ;
    const int inp_h_idx = im_h_idx + filt_h_idx - (filt_h/2) ;
    bool flag = threadlinidx < totalThreads && inp_w_idx >=0 && inp_w_idx <inp_w && inp_h_idx >=0 && inp_h_idx <inp_h;
    if (flag){
        // Calculate Imout Indices

        float randnum;
        float this_px_out=0;
        float float_temp;
        // Flags
       // bool flag = true;
       // bool flag2= false;
       // bool isoutboundh = false;
       // bool isoutbound = false;
        int chan  = __float2int_rn(random[threadlinidx]);
        int input_idx=0;
        //int filt_idx;
        //int j = 0;

        size_t dzdout_idx = calcLinInd4(im_b_idx,filt_n_idx,im_h_idx,im_w_idx ,inp_b, filt_n, inp_h, inp_w);
        size_t dzdin_idx = calcLinInd4(im_b_idx , chan , inp_h_idx , inp_w_idx , inp_b, inp_c, inp_h, inp_w);
//       printf("%d \n",chan);
        //size_t dzdin_idx = im_b_idx + chan + inp_h_idx + inp_w_idx;
        size_t dzdl_filt_idx = calcLinInd4(filt_n_idx , chan , filt_h_idx , filt_w_idx, filt_n, inp_c, filt_h, filt_w );
        float dzdoutthis = dzdout[dzdout_idx];
//dzdin[0] += dzdoutthis;
        //dzdin[dzdin_idx] += dzdoutthis;
        atomicAdd(&(dzdin[dzdin_idx]), dzdoutthis);
        atomicAdd(&(dzdl_filt[dzdl_filt_idx]), dzdoutthis* (input[dzdin_idx] - 1 - lfilt[dzdl_filt_idx])); // p differentiable grad
        //atomicAdd(&(dzdl_filt[dzdl_filt_idx]), -dzdoutthis); // p NON-differentiable grad
        //dzdl_filt[dzdl_filt_idx] += dzdoutthis* input[dzdin_idx] ;
    }



}

template <typename scalar_t>
__global__ void klconv_cuda_backward_kernel(
    const scalar_t* __restrict__ input, //TODO: MAKE sure the dims are dzdin and the threads are compatible
    const scalar_t* __restrict__ lfilt,
    const scalar_t* __restrict__ dzdout,
    const scalar_t* __restrict__ random,
    float* __restrict__ dzdin,
    float* __restrict__ dzdl_filt,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const int totalThreads
){
    const size_t threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    int int_temp_reg;
    const int im_w_idx = threadlinidx % inp_w;
    int_temp_reg =inp_w;
    const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
    int_temp_reg *=inp_h;
    const int filt_n_idx = (threadlinidx/(int_temp_reg)) % filt_n;
    int_temp_reg *=filt_n;
    const int im_b_idx = (threadlinidx/(int_temp_reg)) % inp_b;
    int_temp_reg *= inp_b;
    const int filt_w_idx = (threadlinidx/(int_temp_reg)) % filt_w;
    int_temp_reg *= filt_w;
    const int filt_h_idx = (threadlinidx/(int_temp_reg)) ;

    const int inp_w_idx = im_w_idx + filt_w_idx - (filt_w/2) ;
    const int inp_h_idx = im_h_idx + filt_h_idx - (filt_h/2) ;
    bool flag = threadlinidx < totalThreads && inp_w_idx >=0 && inp_w_idx <inp_w && inp_h_idx >=0 && inp_h_idx <inp_h;
    if (flag){
        // Calculate Imout Indices

        float randnum;
        float this_px_out=0;
        float float_temp;
        int chan  = __float2int_rn(random[threadlinidx]);
        int input_idx=0;
        float dzdoutthis;
        float cur_lfilt;
        float cur_pfilt;
        float cur_in;

        for (chan =0 ; chan < inp_c ; chan++){
            size_t dzdout_idx = calcLinInd4(im_b_idx,filt_n_idx,im_h_idx,im_w_idx ,inp_b, filt_n, inp_h, inp_w);
            size_t dzdin_idx = calcLinInd4(im_b_idx , chan , inp_h_idx , inp_w_idx , inp_b, inp_c, inp_h, inp_w);
            size_t dzdl_filt_idx = calcLinInd4(filt_n_idx , chan , filt_h_idx , filt_w_idx, filt_n, inp_c, filt_h, filt_w );

            cur_lfilt =  lfilt[dzdl_filt_idx];
            cur_pfilt = expf(cur_lfilt);
            cur_in = input[dzdin_idx];

            dzdoutthis = dzdout[dzdout_idx];
            atomicAdd(&(dzdin[dzdin_idx]), (dzdoutthis*cur_pfilt));
            atomicAdd(&(dzdl_filt[dzdl_filt_idx]), dzdoutthis * cur_pfilt* (cur_in - 1 - cur_lfilt)); // p differentiable grad
        }
    }



}


template <typename scalar_t>
__global__ void klconvs_cuda_backward_rand_kernel(
    const scalar_t* __restrict__ input, //TODO: MAKE sure the dims are dzdin and the threads are compatible
    const scalar_t* __restrict__ lfilt,
    const scalar_t* __restrict__ dzdout,
    const scalar_t* __restrict__ random,
    float* __restrict__ dzdin,
    float* __restrict__ dzdl_filt,
    const size_t  filt_h,
    const size_t filt_w,
    const size_t filt_n,
    const size_t inp_h,
    const size_t inp_w,
    const size_t inp_c,
    const size_t inp_b,
    const int totalThreads
){
    const size_t threadlinidx = blockIdx.x*blockDim.x + threadIdx.x;
    int int_temp_reg;
    const int im_w_idx = threadlinidx % inp_w;
    int_temp_reg =inp_w;
    const int im_h_idx = (threadlinidx/int_temp_reg) % inp_h;
    int_temp_reg *=inp_h;
    const int filt_n_idx = (threadlinidx/(int_temp_reg)) % filt_n;
    int_temp_reg *=filt_n;
    const int im_b_idx = (threadlinidx/(int_temp_reg)) % inp_b;
    int_temp_reg *= inp_b;
    const int filt_w_idx = (threadlinidx/(int_temp_reg)) % filt_w;
    int_temp_reg *= filt_w;
    const int filt_h_idx = (threadlinidx/(int_temp_reg)) ;

    const int inp_w_idx = im_w_idx + filt_w_idx - (filt_w/2) ;
    const int inp_h_idx = im_h_idx + filt_h_idx - (filt_h/2) ;
    bool flag = threadlinidx < totalThreads && inp_w_idx >=0 && inp_w_idx <inp_w && inp_h_idx >=0 && inp_h_idx <inp_h;
    if (flag){
        // Calculate Imout Indices

        float randnum;
        float this_px_out=0;
        float float_temp;
        randnum  = random[threadlinidx];
        int chan_idx=0;
        int chan=0;
        float dzdoutthis;
        float cur_lfilt;
        float cur_pfilt;
        float cur_in;
        size_t dzdout_idx = calcLinInd4(im_b_idx,filt_n_idx,im_h_idx,im_w_idx ,inp_b, filt_n, inp_h, inp_w);
        size_t dzdl_filt_idx;
        dzdoutthis = dzdout[dzdout_idx];
        for (chan =0 ; chan < inp_c ; chan++){
            dzdl_filt_idx = calcLinInd4(filt_n_idx , chan , filt_h_idx , filt_w_idx, filt_n, inp_c, filt_h, filt_w );
            cur_lfilt =  lfilt[dzdl_filt_idx];
            cur_pfilt = expf(cur_lfilt);
            if (cur_pfilt >= randnum){
                chan_idx = chan;
                break;
            }
            randnum = randnum - cur_pfilt;

        }
            size_t dzdin_idx = calcLinInd4(im_b_idx , chan_idx , inp_h_idx , inp_w_idx , inp_b, inp_c, inp_h, inp_w);
            cur_in = input[dzdin_idx];
            atomicAdd(&(dzdin[dzdin_idx]), (dzdoutthis));
            atomicAdd(&(dzdl_filt[dzdl_filt_idx]), dzdoutthis * (cur_in - 1 - cur_lfilt)); // p differentiable grad

    }



}

// End Kernels *************************

//Forward wrapper ----------------------
std::vector<at::Tensor> klconvs_cuda_forward(
at::Tensor input,
at::Tensor log_filt){

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
        int j = 32;
        const  int threadsperblock =j*32;
        int blockNum = (totalThreads/threadsperblock);
        if (totalThreads%threadsperblock != 0 ){
            blockNum++;

        }
        const dim3 blocks(blockNum);
        //printf("blocks: %d, totaltherads/threadperbloc : %d", blocks,totalThreads/threadsperblock);

        AT_DISPATCH_FLOATING_TYPES(input.type(), "klconvs_forward_cuda", ([&] {
           klconvs_cuda_forward_kernel<scalar_t><<<blocks, threadsperblock>>>(
              input.data<scalar_t>(),
              log_filt.data<float>(),
              out.data<scalar_t>(),
              random.data<scalar_t>(),// rand . data please fix
              filt_width,
              filt_height,
              filt_num,
              im_width,
              im_height,
              im_nchans,
              batch_sz,
              totalOutPx,
              totalThreads

                );
          }));


        //out = out.sum(0);  /// ZEro Loop Version \TODO: rremove in case of diff kernel
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        return {out,random};

}
//----------------------------------------------
// Backward wrapper
std::vector<at::Tensor> klconvs_cuda_backward(at::Tensor dzdout,
at::Tensor input,
at::Tensor log_filt,
at::Tensor random
){

const auto batch_sz = input.size(0);
const auto im_height = input.size(2);
const auto im_width = input.size(3);
const auto im_nchans = input.size(1);

const auto filt_num = log_filt.size(0);
const auto filt_height = log_filt.size(2);
const auto filt_width = log_filt.size(3);

auto dzdinput = at::zeros_like(input);
auto dzdlfilt = at::zeros_like(log_filt);


// Single Loop const auto totalThreads = totalOutPx*filt_height*filt_width;
const  int totalThreads = im_height*im_width*batch_sz*filt_num*filt_height*filt_width;
int j = 32;
const  int threadsperblock =j*32;
int blockNum = (totalThreads/threadsperblock);
if (totalThreads%threadsperblock != 0 ){
    blockNum++;
}
const dim3 blocks(blockNum);
//printf("blocks: %d, totaltherads/threadperbloc : %d", blocks,totalThreads/threadsperblock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "klconvs_backward_cuda", ([&] {
       klconvs_cuda_backward_kernel<scalar_t><<<blocks, threadsperblock>>>(  // TODO : CHANGE KLCONVS AND KLCONV BACK AND FORTH. NO FORGET.... NEVER FORGET, it is easy not to seee.
          input.data<scalar_t>(),
          log_filt.data<scalar_t>(),
          dzdout.data<scalar_t>(),
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
std::vector<at::Tensor> klconvs_cuda_backward_rand(at::Tensor dzdout,
at::Tensor input,
at::Tensor log_filt
){

const auto batch_sz = input.size(0);
const auto im_height = input.size(2);
const auto im_width = input.size(3);
const auto im_nchans = input.size(1);

const auto filt_num = log_filt.size(0);
const auto filt_height = log_filt.size(2);
const auto filt_width = log_filt.size(3);

auto dzdinput = at::zeros_like(input);
auto dzdlfilt = at::zeros_like(log_filt);

auto random = at::rand(input.type(),{filt_height,filt_width,batch_sz,filt_num,im_height,im_width});
// Single Loop const auto totalThreads = totalOutPx*filt_height*filt_width;
const  int totalThreads = im_height*im_width*batch_sz*filt_num*filt_height*filt_width;
int j = 32;
const  int threadsperblock =j*32;
int blockNum = (totalThreads/threadsperblock);
if (totalThreads%threadsperblock != 0 ){
    blockNum++;
}
const dim3 blocks(blockNum);
//printf("blocks: %d, totaltherads/threadperbloc : %d", blocks,totalThreads/threadsperblock);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "klconvs_backward_rand_cuda", ([&] {
       klconvs_cuda_backward_rand_kernel<scalar_t><<<blocks, threadsperblock>>>(  // TODO : CHANGE KLCONVS AND KLCONV BACK AND FORTH. NO FORGET.... NEVER FORGET, it is easy not to seee.
          input.data<scalar_t>(),
          log_filt.data<scalar_t>(),
          dzdout.data<scalar_t>(),
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