extern "C"
__global__ void conv_2d(int *conf, float *x, float *w, float *bw, float *y)
{
    // TODO
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i<n)
    // {
    //     sum[i] = a[i] + b[i];
    // }
}

__global__ void weights_gradients(int *conf, float *x, float *g, float *y)
{
    // TODO
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i<n)
    // {
    //     sum[i] = a[i] + b[i];
    // }
}

__global__ void inputs_gradients(int *conf, float *w, float *g, float *y)
{
    // TODO
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i<n)
    // {
    //     sum[i] = a[i] + b[i];
    // }
}
