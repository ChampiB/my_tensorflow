// Shared memory.
extern __shared__ float sdata[];

/**
 * Apply the relu function.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void relu(int n, float *x)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        if (x[i] < 0) x[i] = 0;
    }
}

/**
 * Apply the sigmoid function.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void sigmoid(int n, float *x)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        x[i] = 1 / (1 + expf(-x[i]));
    }
}

/**
 * Apply the softmax function.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
__device__ void softMaxDevice(int n, float *x) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = x[index];
    __syncthreads();

    float sum = 0;
    for (int i = 0; i < blockDim.x; i++) {
        sum += expf(sdata[i]);
    }
    x[index] = expf(sdata[threadIdx.x]) / sum;
}

/**
 * Apply the softmax function.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void softMax(int n, float *x)
{
    softMaxDevice(n, x);
}

/**
 * Compute the linear derivative.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void noneDerivative(int n, float *x)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        x[i] = 1;
    }
}

/**
 * Compute the relu derivative.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void reluDerivative(int n, float *x)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        x[i] = (x[i] <= 0) ? 0 : 1;
    }
}

/**
 * Compute the softmax derivative.
 * @param n the number of elements.
 * @param x the data, i.e. the output buffer.
 * @param g the gradient.
 * @return nothing.
 */
extern "C"
__global__ void softMaxDerivative(int n, float *x, float *g)
{
    softMaxDevice(n, x);
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = x[index];
    __syncthreads();

    float sum = 0;
    for (int i = 0; i < blockDim.x; i++) {
        int g_index = blockIdx.x * blockDim.x + i;
        sum += sdata[i] * ((i == threadIdx.x ? 1 : 0) - sdata[threadIdx.x]) * g[g_index];
    }
    x[index] = sum;
}

/**
 * Compute the sigmoid derivative.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void sigmoidDerivative(int n, float *x)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float sig = 1.0 / (1.0 + expf(-x[i]));
        x[i] = sig * (1.0 - sig);
    }
}
