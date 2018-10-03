#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

//
__global__ void quantize_kernel(float *x, int n, float *quantize,int kbits)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    static int integer1,integer2;
    integer1 = 1 << kbits;

    static float temp,temp1,temp2;
    temp=x[i];
    if(temp>1)         quantize[i] = 1;
    else if(temp<-1)   quantize[i] = -1;
    else
    {
       temp1 = (temp + 1) / 2.0;
       temp2 = temp1 * (integer1 - 1);
       integer2 = temp2 + 0.5;
       quantize[i] = (2.0 / (integer1 - 1) * integer2 - 1);
    }
}

void quantize_gpu(float *x, int n, float *quantize,int kbits)
{
    quantize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, quantize,kbits);
    check_error(cudaPeekAtLastError());
}

__global__ void quantize_kernel_second(float *x, int n, float *quantize,int kbits)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    static int integer1,integer2;
    integer1 = 1 << kbits;

    static float temp,temp1,temp2;
    temp = x[i];
    if(temp>1)         quantize[i] = 1;
    else if(temp<-1)   quantize[i] = -1;
    else
    {
       temp1 = temp * integer1;
       integer2 = temp1;
       quantize[i] = (float) integer2 / (float) integer1;
    }
}

void quantize_gpu_second_method(float *x, int n, float *quantize,int kbits)
{
    quantize_kernel_second<<<cuda_gridsize(n), BLOCK>>>(x, n, quantize,kbits);
    check_error(cudaPeekAtLastError());
}
//

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

//
__global__ void three_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;

    int i = 0;
    float mean = 0;
    int count=0;
    float sum=0;

    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean =mean / size;
    mean *= 0.7;


    for(i = 0; i < size; ++i){
        if( fabsf( weights[f*size + i] ) >= mean)
        {
            sum += fabsf(weights[f*size + i]);
            count++;
        }
    }

    sum = sum / count;


    for(i = 0; i < size; ++i){
       if(weights[f*size + i] >= mean)
          binary[f*size + i]=sum;
       else if(weights[f*size + i] <= -mean)
          binary[f*size + i]=-sum;
       else
          binary[f*size + i]=0;
    }

}

void three_weights_gpu(float *weights, int n, int size, float *binary)
{
    three_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}
//

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}



void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    int lgy,temp,lll;
    int number_of_weights = l.nweights;
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    //
    if(l.three){
        three_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.three_weights_gpu);
        swap_three(&l);
    }
    //

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

    //
    if(l.quantize)
    {
       three_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.three_weights_gpu);
       quantize_gpu_second_method(net.input_gpu, l.c*l.h*l.w*l.batch, l.quantize_input_gpu,7);
       swap_three(&l);
       net.input_gpu = l.quantize_input_gpu;
    }
    //


    /*
    quantize_gpu_second_method(net.input_gpu, l.c*l.h*l.w*l.batch, net.input_gpu,3);
    cuda_pull_array(net.input_gpu, net.input, l.inputs);
    for(lgy=0;lgy<100;lgy++)
    {
       printf("%10f",net.input[lgy]);
       if((lgy+1)%10==0) printf("\n");
    }
    printf("Input an integer number:");
    scanf("%d",&lll);
    */


    //
    if(l.print_weights){
        //three_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.weights_gpu);
        pull_convolutional_layer(l);
        printf("Layer total weights number %d\n",number_of_weights);
        for(lgy=0;lgy<number_of_weights;lgy++)
        {
           printf("%10f",l.weights[lgy]);
           if((lgy+1)%10==0) printf("\n");
        }
        printf("\n\n");
    }
    //

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;

            im2col_gpu(net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
                l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //
    /*if(l.print_activations){
      cuda_pull_array(l.output_gpu, l.output, l.outputs);
      FILE *fp;
      fp=fopen("~/test/"+video_name+"_"+frame_index+".txt","w");
      fp=fopen("a.txt","w");
      frame_index++;
      for (lgy = 0; lgy < l.outputs; lgy++) {  //将数组中的整数写入fp指向的txt文件
        fprintf(fp,"%10f\n",l.output[lgy]);
      }
      fclose(fp);*/
       // printf("Layer outputs total number:%d\n",l.outputs);
       // scanf("%d",&temp);
       // cuda_pull_array(l.output_gpu, l.output, l.outputs);
       // for(lgy=0;lgy<l.outputs;lgy++)
       // {
       //     printf("%10f",l.output[lgy]);
       //     if((lgy+1)%10==0) printf("\n");
       // }
       // printf("\n\n");
    }
    //

    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
    //
    if(l.three || l.quantize) swap_three(&l);
    //
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
    //
    if(l.quantize) net.input_gpu = l.quantize_input_gpu;
    //
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        //
        if(l.three || l.quantize) swap_three(&l);
        //
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        //
        if(l.three || l.quantize) swap_three(&l);
        //
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
        //
        if(l.quantize) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
        //
    }

#else
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w,
                    l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if(net.delta_gpu){
                if(l.binary || l.xnor) swap_binary(&l);
                //
                if(l.three || l.quantize) swap_three(&l);
                //
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride,
                    l.pad, net.delta_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w);
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
                //
                if(l.three  ||  l.quantize) swap_three(&l);
                //
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
            //
            if(l.quantize) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
            //
        }
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}
