/**
 * Compute the padding activation.
 * @param x_shape the shape of the input buffer.
 * @param x the input buffer.
 * @param y_shape the shape of the output buffer.
 * @param y the output buffer.
 * @param value the padding value.
 * @return nothing.
 */
extern "C"
__global__ void activation(long *x_shape, float *x, long *y_shape, float *y, float value)
{
    int yfs = y_shape[2] * y_shape[3]; // Y feature size.
    int y_index = threadIdx.x * y_shape[1] * yfs + blockIdx.x * yfs + blockIdx.y * y_shape[3] + blockIdx.z;
    if (threadIdx.x >= x_shape[0] || blockIdx.x >= x_shape[1] || blockIdx.y >= x_shape[2] || blockIdx.z >= x_shape[3]) {
        y[y_index] = value;
    } else {
        int xfs = x_shape[2] * x_shape[3]; // X feature size.
        int x_index = threadIdx.x * x_shape[1] * xfs + blockIdx.x * xfs + blockIdx.y * x_shape[3] + blockIdx.z;
        y[y_index] = x[x_index];
    }
}
// (int)ycShape[1], (int)ycShape[2], (int)ycShape[3],
// (int)ycShape[0], 1, 1,
/**
 * Compute the gradients with respect to the inputs.
 * @param x_shape the shape of the input buffer.
 * @param x the input buffer.
 * @param y_shape the shape of the output buffer.
 * @param y the output buffer.
 * @return nothing.
 */
extern "C"
__global__ void inputs_gradients(long *x_shape, float *x, long *y_shape, float *y)
{
    if (threadIdx.x < y_shape[0] && blockIdx.x < y_shape[1] && blockIdx.y < y_shape[2] && blockIdx.z < y_shape[3]) {
        int yfs = y_shape[2] * y_shape[3]; // Y feature size.
        int y_index = threadIdx.x * y_shape[1] * yfs + blockIdx.x * yfs + blockIdx.y * y_shape[3] + blockIdx.y;
        int xfs = x_shape[2] * x_shape[3]; // X feature size.
        int x_index = threadIdx.x * x_shape[1] * xfs + blockIdx.x * xfs + blockIdx.y * x_shape[3] + blockIdx.y;
        y[y_index] = x[x_index];
    }
}
