#include <limits>

/**
 * Configuration indexes.
 */
#define X_IMAGE_SIZE conf[0]
#define X_FEATURE_SIZE conf[1]
#define X_ROW_SIZE conf[2]
#define N conf[3]

/**
 * Compute the index of the maximal value over all the feature maps.
 * @param conf the kernel's configuration.
 * @param x the initial layer activation.
 * @param y the kWTA layer activation, i.e. output buffer.
 * @param i the offset.
 * @return the index of the maximal value.
 */
__device__ int arg_max(int *conf, float *x, float *y, int i) {
    float m = -INFINITY;
    int mi = i;
    for (int j = 0; j < X_IMAGE_SIZE; j += X_FEATURE_SIZE) {
        int index = i + j;
        if (x[index] > m && y[index] == 0) {
            m = x[index];
            mi = index;
        }
    }
    return mi;
}

/**
 * Compute the kWTA activation of the layer.
 * @param conf the kernel's configuration.
 * @param k the number of winners.
 * @param x the initial layer activation.
 * @param y the kWTA layer activation, i.e. output buffer.
 * @param m the mask.
 * @return nothing.
 */
extern "C"
__global__ void activation(int *conf, int k, float *x, float *y, float *m)
{
    int index = threadIdx.x * X_IMAGE_SIZE + blockIdx.x * X_ROW_SIZE + blockIdx.y;
    if (index < N) {
        for (int j = 0; j < X_IMAGE_SIZE; j += X_FEATURE_SIZE) {
            y[index + j] = 0;
            m[index + j] = 0;
        }
        for (int j = 0; j < k; j++) {
            int idx = arg_max(conf, x, y, index);
            y[idx] = x[idx];
            m[idx] = 1;
        }
    }
}
