#include <limits>

/**
 * Configuration indexes.
 */
#define KERNEL_0 conf[0]
#define KERNEL_1 conf[1]
#define X_IMAGE_SIZE conf[2]
#define X_FEATURE_SIZE conf[3]
#define X_ROW_SIZE conf[4]
#define N conf[5]
#define Y_IMAGE_SIZE conf[6]
#define Y_FEATURE_SIZE conf[7]
#define Y_ROW_SIZE conf[8]

/**
 * Compute the kernel offset that correspond to the general index i.
 * @param conf is the kernel configuration.
 * @param i is the general index.
 * @return the kernel offset.
 */
__device__ int compute_offset(int *conf, int i) {
    return
        threadIdx.x * X_IMAGE_SIZE +
        blockIdx.x * X_FEATURE_SIZE +
        blockIdx.y * X_ROW_SIZE * KERNEL_0 +
        blockIdx.z * KERNEL_0;
}

/**
 * Compute the maximal value of the kernel corresponding to the kernel offset.
 * @param conf the kernel's configuration.
 * @param x the input activation.
 * @param offset the kernel offset.
 * @return the maximal value.
 */
__device__ float *max(int *conf, float *x, int offset) {
    float *res = x;
    float m = x[0];
    for (int j = 0; j < KERNEL_0; j++) {
        for (int k = 0; k < KERNEL_1; k++) {
            int index = offset + j + k * X_ROW_SIZE;
            if (x[index] > m) {
                m = x[index];
                res = &x[index];
            }
        }
    }
    return res;
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
    for (int j = 0; j < KERNEL_0; j++) {
        for (int k = 0; k < KERNEL_1; k++) {
            int index = offset + j * X_ROW_SIZE + k;
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
__global__ void training_activation(int *conf, float *x, float *r, float *m)
{
    int index = threadIdx.x * Y_IMAGE_SIZE + blockIdx.x * Y_FEATURE_SIZE + blockIdx.y * Y_ROW_SIZE + blockIdx.z;
    if (index < N) {
        int offset = compute_offset(conf, index);
        for (int j = 0; j < KERNEL_0; j++) {
            for (int k = 0; k < KERNEL_1; k++) {
                m[offset + j + k * X_ROW_SIZE] = 0;
            }
        }
        float *mx = max(conf, x, offset);
        r[index] = *mx;
        m[mx - x] = 1;
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
__global__ void activation(int *conf, float *x, float *r)
{
    int index = threadIdx.x * Y_IMAGE_SIZE + blockIdx.x * Y_FEATURE_SIZE + blockIdx.y * Y_ROW_SIZE + blockIdx.z;
    if (index < N) {
        r[index] = *max(conf, x, compute_offset(conf, index));
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
    int index = threadIdx.x * Y_IMAGE_SIZE + blockIdx.x * Y_FEATURE_SIZE + blockIdx.y * Y_ROW_SIZE + blockIdx.z;
    if (index < N) {
        mul(conf, m, compute_offset(conf, index), g[index]);
    }
}
