/**
 * Configuration indexes.
 */
#define Y_IMAGE_SIZE conf[0]
#define Y_FEATURE_SIZE conf[1]
#define Y_ROW_SIZE conf[5]
#define Y_0 conf[2]
#define Y_1 conf[3]
#define Y_2 conf[4]
#define Y_3 conf[5]
#define STRIDES_0 conf[6]
#define STRIDES_1 conf[7]
#define W_0 conf[8]
#define W_1 conf[9]
#define W_2 conf[10]
#define W_3 conf[11]
#define X_IMAGE_SIZE conf[12]
#define X_CHANNEL_SIZE conf[13]
#define X_ROW_SIZE conf[14]

/**
 * The memory shared between the threads of each block.
 */
extern __shared__ float sdata[];

/**
 * Compute the sum of the array's elements.
 * @param sdata the array.
 * @return the sum.
 */
__device__ float reduce_sum(float *sdata) {
    for (int i = 1; i < blockDim.x * blockDim.y * blockDim.z; i++) {
        sdata[0] += sdata[i];
    }
    return sdata[0];
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
    int bid = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    int tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;

    int fi = blockIdx.x / W_1;
    int4 w_pos = make_int4(fi, blockIdx.x - fi * W_1, blockIdx.y, blockIdx.z);
    int4 y_pos = make_int4(threadIdx.x, fi, threadIdx.y, threadIdx.z);

    sdata[tid] = 0;
    for (int i = y_pos.x; i < Y_0; i += blockDim.x) {
        for (int r = y_pos.z; r < Y_2; r += blockDim.y) {
            for (int c = y_pos.w; c < Y_3; c += blockDim.z) {
                int rx = r * STRIDES_0 + w_pos.z;
                int cx = c * STRIDES_1 + w_pos.w;
                int x_index = i * X_IMAGE_SIZE + w_pos.y * X_CHANNEL_SIZE + rx * X_ROW_SIZE + cx;
                int y_index = i * Y_IMAGE_SIZE + w_pos.x * Y_FEATURE_SIZE + r * Y_ROW_SIZE + c;
                sdata[tid] += min(y[y_index], (float)1) * (x[x_index] - w[bid]);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        r[bid] = reduce_sum(sdata);
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
    int tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
    sdata[tid] = 0;
    int index = threadIdx.x * Y_IMAGE_SIZE + blockIdx.x * Y_FEATURE_SIZE + threadIdx.y * Y_ROW_SIZE + threadIdx.z;
    for (int vo = 0; threadIdx.y + vo < Y_2; vo += blockDim.y) {
        for (int ho = 0; threadIdx.z + ho < Y_3; ho += blockDim.z) {
            int y_index = index + vo * Y_3 + ho;
            if (y[y_index] != 0)
                sdata[tid] += 1;
        }
    }
    __syncthreads();

    if (tid == 0) {
        r[blockIdx.x] = reduce_sum(sdata);
    }
}
