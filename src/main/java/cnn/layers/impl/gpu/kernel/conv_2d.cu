/**
 * Configuration indexes.
 */
#define FILTERS_0 conf[0]
#define FILTERS_1 conf[1]
#define FILTERS_2 conf[2]
#define STRIDES_0 conf[3]
#define STRIDES_1 conf[4]
#define X_1 conf[5]
#define X_2 conf[6]
#define X_3 conf[7]
#define N conf[8]
#define Y_0 conf[9]
#define Y_1 conf[10]
#define Y_2 conf[11]
#define Y_3 conf[12]
#define W_0 conf[13]
#define W_1 conf[14]
#define W_2 conf[15]
#define W_3 conf[16]
#define Y_IMAGE_SIZE conf[17]
#define Y_FEATURE_SIZE conf[18]
#define Y_ROW_SIZE conf[19]
#define X_IMAGE_SIZE conf[20]
#define X_CHANNEL_SIZE conf[21]
#define X_ROW_SIZE conf[22]

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
__device__ int4 position_of(int i, int3 in)
{
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
__device__ int index_of(int4 pos, int3 in)
{
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
__device__ float reduce_sum(float *sdata)
{
    for (int i = 1; i < blockDim.x * blockDim.y * blockDim.z; i++) {
        sdata[0] += sdata[i];
    }
    return sdata[0];
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
    int index = threadIdx.x * Y_IMAGE_SIZE + blockIdx.x * Y_FEATURE_SIZE + blockIdx.y * Y_ROW_SIZE + blockIdx.z;
    if (index < N) {
        y[index] = 0;
        int x_offset = threadIdx.x * X_1 * X_2 * X_3 + blockIdx.y * STRIDES_0 * X_3 + blockIdx.z * STRIDES_1;
        int w_offset = blockIdx.x * X_1 * FILTERS_1 * FILTERS_2;
        for (int j = 0; j < X_1; j++) {
            for (int k = 0; k < FILTERS_1; k++) {
                for (int l = 0; l < FILTERS_2; l++) {
                    int x_index = x_offset + (j * X_2 + k) * X_3 + l;
                    int w_index = w_offset + (j * FILTERS_1 + k) * FILTERS_2 + l;
                    y[index] += x[x_index] * w[w_index];
                }
            }
        }
        y[index] += bw[blockIdx.x];
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
    int bid = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    int tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;

    int fi = blockIdx.x / W_1;
    int4 w_pos = make_int4(fi, blockIdx.x - fi * W_1, blockIdx.y, blockIdx.z);

    sdata[tid] = 0;
    for (int i = threadIdx.x; i < Y_0; i += blockDim.x) {
        for (int r = threadIdx.y; r < Y_2; r += blockDim.y) {
            for (int c = threadIdx.z; c < Y_3; c += blockDim.z) {
                int rx = r * STRIDES_0 + w_pos.z;
                int cx = c * STRIDES_1 + w_pos.w;
                int x_index = i * X_IMAGE_SIZE + w_pos.y * X_CHANNEL_SIZE + rx * X_ROW_SIZE + cx;
                int g_index = i * Y_IMAGE_SIZE + w_pos.x * Y_FEATURE_SIZE + r * Y_ROW_SIZE + c;
                sdata[tid] += g[g_index] * x[x_index];
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        r[bid] = reduce_sum(sdata);
    }
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
__device__ float compute_sum_of_gradients(int *conf, int4 x_pos, int3 g_shape, float *g, int3 w_shape, float *w)
{
    float gradient = 0;
    for (int j = 0; j < FILTERS_0; j++) {
        int4 g_pos = make_int4(x_pos.x, j, x_pos.z - FILTERS_1 + 1, x_pos.w - FILTERS_2 + 1);
        for (int k = 0; k < FILTERS_1; k++) {
            for (int l = 0; l < FILTERS_2; l++) {
                if (
                    g_pos.x >= 0 && g_pos.x < Y_0 &&
                    g_pos.z >= 0 && g_pos.z < Y_2 &&
                    g_pos.w >= 0 && g_pos.w < Y_3
                ) {
                    int4 w_pos = make_int4(j, x_pos.y, FILTERS_1 - 1 - k, FILTERS_2 - 1 - l);
                    gradient += w[index_of(w_pos, w_shape)] * g[index_of(g_pos, g_shape)];
                }
                g_pos.w++;
            }
            g_pos.w -= FILTERS_2;
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
    int3 g_shape = make_int3(Y_1, Y_2, Y_3);
    int3 w_shape = make_int3(W_1, W_2, W_3);
    int3 x_shape = make_int3(X_1, X_2, X_3);
    for (int i = index; i <= N; i += stride) {
        int4 x_pos = position_of(i, x_shape);
        r[i] = compute_sum_of_gradients(conf, x_pos, g_shape, g, w_shape, w);
    }
}
