#include <limits>

/**
 * Configuration indexes.
 */
#define KERNEL_0 0
#define KERNEL_1 1
#define X_1 2
#define X_2 3
#define X_3 4
#define N 5

/**
 * Compute the kernel offset that correspond to the general index i.
 * @param conf is the kernel configuration.
 * @param i is the general index.
 * @return the kernel offset.
 */
__device__ int compute_offset(int *conf, int i) {
    int rs = conf[X_3] / conf[KERNEL_1];
    int fs = (conf[X_2] / conf[KERNEL_0]) * rs;
    int is = (conf[X_2] / conf[KERNEL_0]) * (conf[X_3] / conf[KERNEL_1]) * conf[X_1];
    int ii = i / is;
    int fi = (i - ii * is) / fs;
    int ri = (i - ii * is - fi * fs) / rs;
    int ci = i - ii * is - fi * fs - ri * rs;
    return
        ii * conf[X_1] * conf[X_2] * conf[X_3] +
        fi * conf[X_2] * conf[X_3] +
        ri * conf[X_3] * conf[KERNEL_0] +
        ci * conf[KERNEL_0];
}

/**
 * Compute the maximal value of the kernel corresponding to the kernel offset.
 * @param conf the kernel's configuration.
 * @param x the input activation.
 * @param offset the kernel offset.
 * @return the maximal value.
 */
__device__ float max(int *conf, float *x, int offset) {
    float m = -INFINITY;
    for (int j = 0; j < conf[KERNEL_0]; j++) {
        for (int k = 0; k < conf[KERNEL_1]; k++) {
            int index = offset + j + k * conf[X_3];
            if (x[index] > m)
                m = x[index];
        }
    }
    return m;
}

/**
 * Compute the index of the maximal value of the kernel corresponding to the kernel offset.
 * @param conf the kernel's configuration.
 * @param x the input activation.
 * @param offset the kernel offset.
 * @return the index of the maximal value.
 */
__device__ int arg_max(int *conf, float *x, int offset) {
    float m = -INFINITY;
    int mi = 0;
    for (int j = 0; j < conf[KERNEL_0]; j++) {
        for (int k = 0; k < conf[KERNEL_1]; k++) {
            int index = offset + j + k * conf[X_3];
            if (x[index] > m) {
                m = x[index];
                mi = index;
            }
        }
    }
    return mi;
}

/**
 * Multiply the kernel corresponding to the kernel offset by n.
 * @param conf the kernel's configuration.
 * @param x the input activation.
 * @param offset the kernel offset.
 * @param n the multiplication value.
 * @return nothing.
 */
__device__ void mul(int *conf, float *x, int offset, float n) {
    for (int j = 0; j < conf[KERNEL_0]; j++) {
        for (int k = 0; k < conf[KERNEL_1]; k++) {
            int index = offset + j * conf[X_3] + k;
            x[index] = x[index] * n;
        }
    }
}

/**
 * Compute the activation of the max pooling layer and the pooling mask.
 * @param conf the kernel's configuration.
 * @param x the input activation.
 * @param r the layer output activation, i.e. output buffer.
 * @param m the pooling mask, i.e. output buffer.
 * @return nothing.
 */
extern "C"
__global__ void training_max_pooling_2d(int *conf, float *x, float *r, float *m)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < conf[N]; i += stride) {
        int offset = compute_offset(conf, i);
        r[i] = max(conf, x, offset);
        m[arg_max(conf, x, offset)] = 1;
    }
}

/**
 * Compute the activation of the max pooling layer.
 * @param conf the kernel's configuration.
 * @param x the input activation.
 * @param r the layer output activation, i.e. output buffer.
 * @return nothing.
 */
extern "C"
__global__ void max_pooling_2d(int *conf, float *x, float *r)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < conf[N]; i += stride) {
        int offset = compute_offset(conf, i);
        r[i] = max(conf, x, offset);
    }
}

/**
 * Compute the gradient with respect to the inputs.
 * @param conf the kernel's configuration.
 * @param g the gradient with respect to the outputs.
 * @param m the pooling mask, i.e. output buffer.
 * @return nothing.
 */
extern "C"
__global__ void inputs_gradients(int *conf, float *g, float *m)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < conf[N]; i += stride) {
        int offset = compute_offset(conf, i);
        mul(conf, m, offset, g[i]);
    }
}
