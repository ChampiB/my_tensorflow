/**
 * Configuration indexes.
 */
#define FILTERS_0 0
#define FILTERS_1 1
#define FILTERS_2 2
#define STRIDES_0 3
#define STRIDES_1 4
#define X_1 5
#define X_2 6
#define X_3 7
#define N 8
#define Y_0 9
#define Y_1 10
#define Y_2 11
#define Y_3 12
#define W_0 13
#define W_1 14
#define W_2 15
#define W_3 16

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
__device__ int4 position_of(int i, int3 in) {
    int sy = in.y * in.z;
    int sx = in.x * sy;
    int x = i / sx;
    int y = (i - x * sx) / sy;
    int z = (i - x * sx - y * sy) / in.z;
    int w = i - x * sx - y * sy - z * in.z;
    return make_int4(x, y, z, w);
}

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
 * Compute the position in y that correspond to the general index i.
 * @param conf is the kernel configuration.
 * @param i is the general index.
 * @return the position in y.
 */
__device__ int4 compute_activation_y_position(int *conf, int i) {
    int fs = conf[Y_2] * conf[Y_3];
    int is = fs * conf[FILTERS_0];
    int ii = i / is;
    int fi = (i - ii * is) / fs;
    int ri = (i - ii * is - fi * fs) / conf[Y_3];
    int ci = i - ii * is - fi * fs - ri * conf[Y_3];
    return make_int4(ii, fi, ri, ci);
}

/**
 * Compute the gradient position corresponding to the general index i.
 * @conf the kernel's configuration.
 * @i the general index.
 * @return the gradient position.
 */
__device__ int4 compute_gradient_position(int *conf, int i) {
    int fs = conf[Y_2] * conf[Y_3];
    int is = conf[Y_1] * fs;
    int ii = i / is;
    int fi = blockIdx.x / (conf[W_1] * conf[W_2] * conf[W_3]);
    int ri = (i - ii * is) / conf[Y_3];
    int ci = i - ii * is - ri * conf[Y_3];
    return make_int4(ii, fi, ri, ci);
}

/**
 * Compute the x offset corresponding to the general index.
 * @param conf the kernel's configuration.
 * @param y_pos the position in y.
 * @return the x offset.
 */
__device__ int compute_x_offset(int *conf, int4 y_pos) {
    return
        y_pos.x * conf[X_1] * conf[X_2] * conf[X_3] +
        y_pos.z * conf[STRIDES_0] * conf[X_3] +
        y_pos.w * conf[STRIDES_1];
}

/**
 * Compute the neuron activation.
 * @param conf is the kernel's configuration.
 * @param i is the general index.
 * @param x is the layer inputs.
 * @param w is the layers' weights.
 * @param wb is the layers' bias weights.
 * @return the neuron activation.
 */
__device__ float neuron_activation(int *conf, int i, float *x, float *w, float *wb) {
    int4 y_pos = compute_activation_y_position(conf, i);
    int x_offset = compute_x_offset(conf, y_pos);
    int w_offset = y_pos.y * conf[X_1] * conf[FILTERS_1] * conf[FILTERS_2];
    float y = 0;
    for (int j = 0; j < conf[X_1]; j++) {
        for (int k = 0; k < conf[FILTERS_1]; k++) {
            for (int l = 0; l < conf[FILTERS_2]; l++) {
                int x_index = x_offset + (j * conf[X_2] + k) * conf[X_3] + l;
                int w_index = w_offset + (j * conf[FILTERS_1] + k) * conf[FILTERS_2] + l;
                y += x[x_index] * w[w_index];
            }
        }
    }
    return y + wb[y_pos.y];
}

/**
 * Compute the x index.
 * @param conf is the kernel' configuration.
 * @param y_pos is the position in y corresponding to the general index i.
 * @return the index.
 */
__device__ int compute_x_index(int *conf, int4 g_pos) {
    // Weights' sizes and indexes.
    int wcs = conf[W_2] * conf[W_3];
    int wfs = conf[W_1] * wcs;
    int wfi = blockIdx.x / wfs;
    int wci = (blockIdx.x - wfi * wfs) / wcs;
    int wvi = (blockIdx.x - wfi * wfs - wci * wcs) / conf[W_3];
    int whi = blockIdx.x - wfi * wfs - wci * wcs - wvi * conf[W_3];
    // Input' indexes.
    int ri = g_pos.z * conf[STRIDES_0] + wvi;
    int ci = g_pos.w * conf[STRIDES_1] + whi;
    // Compute the x index and return.
    return index_of(make_int4(g_pos.x, wci, ri, ci), make_int3(conf[X_1], conf[X_2], conf[X_3]));
}

/**
 * Compute the convolution activation.
 * @param conf is the configuration of the kernel.
 * @param x is the input activation.
 * @param w is the weights of the layer.
 * @param bw is the bias weights of the layer.
 * @param y is the output of the layer.
 * @return nothing.
 */
extern "C"
__global__ void activation(int *conf, float *x, float *w, float *bw, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < conf[N]; i += stride) {
        y[i] = neuron_activation(conf, i, x, w, bw);
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
    sdata[threadIdx.x] = 0;
    int n = conf[Y_0] * conf[Y_2] * conf[Y_3];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int4 g_pos = compute_gradient_position(conf, i);
        int g_index = index_of(g_pos, make_int3(conf[Y_1], conf[Y_2], conf[Y_3]));
        int x_index = compute_x_index(conf, g_pos);
        sdata[threadIdx.x] += g[g_index] * x[x_index];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        r[blockIdx.x] = reduce_sum(sdata);
    }
}

/**
 * Compute the position of the top right gradient.
 * @param conf the kernel's configuration.
 * @param j the index of the feature being processed.
 * @param x_pos the position of x.
 * @return the gradient position.
 */
__device__ int4 compute_top_right_gradient_position(int *conf, int j, int4 x_pos) {
    int z = x_pos.z - conf[FILTERS_1] + 1;
    int w = x_pos.w - conf[FILTERS_2] + 1;
    return make_int4(x_pos.x, j, z, w);
}

/**
 * Compute the sum of the gradient over all output unit connected to the position x.
 * @param conf the kernel's configuration.
 * @param x_pos the position of x.
 * @param g_shape the shape of the gradients.
 * @param g the gradients.
 * @param w_shape the shape of the weights.
 * @param w the weights.
 * @return the sum.
 */
__device__ float compute_sum_of_gradients(int *conf, int4 x_pos, int3 g_shape, float *g, int3 w_shape, float *w) {
    float gradient = 0;
    for (int j = 0; j < conf[FILTERS_0]; j++) {
        int4 g_pos = compute_top_right_gradient_position(conf, j, x_pos);
        for (int k = 0; k < conf[FILTERS_1]; k++) {
            for (int l = 0; l < conf[FILTERS_2]; l++) {
                if (
                    g_pos.x >= 0 && g_pos.x < conf[Y_0] &&
                    g_pos.z >= 0 && g_pos.z < conf[Y_2] &&
                    g_pos.w >= 0 && g_pos.w < conf[Y_3]
                ) {
                    int4 w_pos = make_int4(j, x_pos.y, conf[FILTERS_1] - 1 - k, conf[FILTERS_2] - 1 - l);
                    gradient += w[index_of(w_pos, w_shape)] * g[index_of(g_pos, g_shape)];
                }
                g_pos.w++;
            }
            g_pos.w -= conf[FILTERS_2];
            g_pos.z++;
        }
    }
    return gradient;
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int3 g_shape = make_int3(conf[Y_1], conf[Y_2], conf[Y_3]);
    int3 w_shape = make_int3(conf[W_1], conf[W_2], conf[W_3]);
    int3 x_shape = make_int3(conf[X_1], conf[X_2], conf[X_3]);
    for (int i = index; i < conf[N]; i += stride) {
        int4 x_pos = position_of(i, x_shape);
        r[i] = compute_sum_of_gradients(conf, x_pos, g_shape, g, w_shape, w);
    }
}
