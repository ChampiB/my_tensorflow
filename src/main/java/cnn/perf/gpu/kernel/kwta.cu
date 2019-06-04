#include <limits>
#include <stdio.h>

__device__ int compute_offset(int *conf, int i) {
    int is = conf[1] * conf[2];
    int ii = i / is;
    int ri = (i - ii * is) / conf[2];
    int ci = i - ii * is - ri * conf[2];
    return
        ii * conf[0] * conf[1] * conf[2] +
        ri * conf[2] +
        ci;
}

__device__ int arg_max(int *conf, float *x, float *y, int i) {
    float m = -INFINITY;
    int mi = i;
    int stride = conf[1] * conf[2];
    for (int j = 0; j < conf[0] * stride; j += stride) {
        int index = i + j;
        if (x[index] > m && y[index] == 0) {
            m = x[index];
            mi = index;
        }
    }
    return mi;
}

extern "C"
__global__ void activation(int *conf, int k, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < conf[3]; i += stride) {
        int offset = compute_offset(conf, i);
        for (int j = 0; j < k; j++) {
            int idx = arg_max(conf, x, y, offset);
            y[idx] = x[idx];
        }
    }
}
