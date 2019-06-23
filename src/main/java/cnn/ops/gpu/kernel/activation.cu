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
 * Compute the sigmoid derivative.
 * @param n the number of elements.
 * @param x the data.
 * @return nothing.
 */
extern "C"
__global__ void sigmoidDerivative(int n, float *x)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        float sig = 1 / (1 + expf(-x[i]));
        x[i] = sig * (1 - sig);
    }
}
