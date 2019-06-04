/**
 * Configuration indexes.
 */
#define Y_0 0
#define Y_1 1
#define Y_2 2
#define Y_3 3
#define W_1 4
#define W_2 5
#define W_3 6
#define X_0 7
#define X_1 8
#define X_2 9
#define X_3 10
#define STRIDES_0 11
#define STRIDES_1 12

/**
 * The memory shared between the threads of each block.
 */
extern __shared__ float sdata[];

/**
 * Compute the index of the pos in a 4D space:
 * @param pos is the position of the point (i.e. images, features, rows and columns);
 * @param in is the size of the box (i.e. number of features, rows and columns).
 * @return the index of the position.
 */
__device__ int index_of(int4 pos, int3 in) {
    return
        pos.x * in.x * in.y * in.z +
        pos.y * in.y * in.z +
        pos.z * in.z +
        pos.w;
}

/**
 * Compute the sum of the array's elements.
 * @param sdata the array.
 * @return the sum.
 */
__device__ float reduce_sum(float *sdata) {
    float res = 0;
    for (int i = 0; i < blockDim.x; i++) {
        res += sdata[i];
    }
    return res;
}

/**
 * Compute the y index.
 * @param conf is the kernel' configuration.
 * @param i is the general index being processed.
 * @return the index.
 */
__device__ int4 compute_y_position(int *conf, int i) {
    int fs = conf[Y_2] * conf[Y_3];
    int is = conf[Y_1] * fs;
    int ii = i / is;
    int fi = blockIdx.x / (conf[W_1] * conf[W_2] * conf[W_3]);
    int ri = (i - ii * is) / conf[Y_3];
    int ci = i - ii * is - ri * conf[Y_3];
    return make_int4(ii, fi, ri, ci);
}

/**
 * Compute the x index.
 * @param conf is the kernel' configuration.
 * @param y_pos is the position in y corresponding to the general index i.
 * @return the index.
 */
__device__ int compute_x_index(int *conf, int4 y_pos) {
    // Weights' sizes and indexes.
    int wcs = conf[W_2] * conf[W_3];
    int wfs = conf[W_1] * wcs;
    int wfi = blockIdx.x / wfs;
    int wci = (blockIdx.x - wfi * wfs) / wcs;
    int wvi = (blockIdx.x - wfi * wfs - wci * wcs) / conf[W_3];
    int whi = blockIdx.x - wfi * wfs - wci * wcs - wvi * conf[W_3];
    // Input' indexes.
    int ri = y_pos.z * conf[STRIDES_0] + wvi;
    int ci = y_pos.w * conf[STRIDES_1] + whi;
    // Compute the x index and return.
    return index_of(make_int4(y_pos.x, wci, ri, ci), make_int3(conf[X_1], conf[X_2], conf[X_3]));
}

/**
 * Compute the CPCA gradients.
 * @param conf is the kernel' configuration.
 * @param x the input activation.
 * @param w the weights.
 * @param y the output activation.
 * @param r the gradients, i.e. the output buffer.
 * @return nothing.
 */
extern "C"
__global__ void weights_gradients(int *conf, float *x, float *w, float *y, float *r)
{
    sdata[threadIdx.x] = 0;
    int n = conf[0] * conf[2] * conf[3];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int4 y_pos = compute_y_position(conf, i);
        int y_index = index_of(y_pos, make_int3(conf[Y_1], conf[Y_2], conf[Y_3]));
        int x_index = compute_x_index(conf, y_pos);
        sdata[threadIdx.x] += y[y_index] * (x[x_index] - w[blockIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        r[blockIdx.x] = reduce_sum(sdata);
    }
}

/**
 * Count the patterns.
 * @param conf is the kernel' configuration.
 * @param y the output activation.
 * @param r the count, i.e. the output buffer.
 * @return nothing.
 */
extern "C"
__global__ void count_patterns(int *conf, float *y, float *r)
{
    sdata[threadIdx.x] = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n = conf[0] * conf[2] * conf[3];
    for (int i = index; i < n; i += stride) {
        int y_index = index_of(
            compute_y_position(conf, i),
            make_int3(conf[Y_1], conf[Y_2], conf[Y_3])
        );
        if (y[y_index] != 0)
            sdata[threadIdx.x] += 1;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        r[blockIdx.x] = reduce_sum(sdata);
    }
}
