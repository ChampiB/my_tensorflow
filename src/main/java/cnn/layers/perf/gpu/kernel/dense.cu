/**
 * Configuration indexes.
 */
#define NUMBER_OF_IMAGES conf[0]
#define NUMBER_OF_INPUTS conf[1]
#define NUMBER_OF_OUTPUTS conf[2]

/**
 * The memory shared between the threads of each block.
 */
extern __shared__ float sdata[];

/**
 * Compute the sum of the array's elements.
 * @param sdata the array.
 * @return the sum.
 */
__device__ float reduce_sum(float *sdata)
{
    for (int i = 1; i < blockDim.x; i++) {
        sdata[0] += sdata[i];
    }
    return sdata[0];
}

/**
 * Compute the convolution activation.
 * @param conf is the configuration of the kernel.
 * @param x is the input activation.
 * @param w is the weights of the layer.
 * @param y is the output of the layer.
 * @return nothing.
 */
extern "C"
__global__ void activation(int *conf, float *x, float *w, float *y)
{
    int bid = blockIdx.x * NUMBER_OF_OUTPUTS + blockIdx.y;
    sdata[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < NUMBER_OF_INPUTS; i += blockDim.x) {
        sdata[threadIdx.x] += x[blockIdx.x * NUMBER_OF_INPUTS + i] * w[i * gridDim.y + blockIdx.y];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        y[bid] = reduce_sum(sdata);
    }
}

/**
 * Compute the gradients with respect to the weights.
 * @param conf is the configuration of the kernel.
 * @param x is the input activation.
 * @param g is the gradients with respect to the output.
 * @param r is the weights gradients, i.e. output buffer.
 * @return nothing.
 */
extern "C"
__global__ void weights_gradients(int *conf, float *x, float *g, float *r)
{
    int tid = threadIdx.x;
    sdata[tid] = 0;
    for (int ii = threadIdx.x; ii < NUMBER_OF_IMAGES; ii += blockDim.x) {
        sdata[tid] += x[ii * NUMBER_OF_INPUTS + blockIdx.y] * g[ii * NUMBER_OF_OUTPUTS + blockIdx.x];
    }
    __syncthreads();

    if (tid == 0) {
        r[blockIdx.y * NUMBER_OF_OUTPUTS + blockIdx.x] = reduce_sum(sdata);
    }
}

/**
 * Compute the gradients with respect to the weights.
 * @param conf is the configuration of the kernel.
 * @param w is the weights of the layer.
 * @param g is the gradients with respect to the output.
 * @param r is the weights gradients, i.e. output buffer.
 * @return nothing.
 */
extern "C"
__global__ void inputs_gradients(int *conf, float *w, float *g, float *r)
{
    sdata[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < NUMBER_OF_IMAGES; i += blockDim.x) {
        sdata[threadIdx.x] +=
            w[i * NUMBER_OF_INPUTS + blockIdx.y + 1] * g[i * NUMBER_OF_OUTPUTS + blockIdx.y + 1];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        r[blockIdx.y * gridDim.x + blockIdx.x] = reduce_sum(sdata);
    }
}
