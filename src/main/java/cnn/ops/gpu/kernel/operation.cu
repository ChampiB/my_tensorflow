/**
 * The memory shared between the threads of each block.
 */
extern __shared__ float sdata[];

/**
 * Arg max function along a row.
 * @param n the number of column.
 * @param i the row index.
 * @param a the array (data).
 * @param r the output buffer.
 * @return nothing.
 */
extern "C"
__global__ void arg_max_row(int n, int i, float *a, float *r)
{
    int offset = n * i;
    float m = a[offset];
    float mi = 0;
    for (int j = 1; j < n; j++) {
        if (m < a[offset + j]) {
            m = a[offset + j];
            mi = (float)j;
        }
    }
    r[0] = mi;
}

/**
 * Sum the elements of the array.
 * @param n the number of elements.
 * @param a the array (data).
 * @param r the output buffer.
 * @return nothing.
 */
extern "C"
__global__ void sum(int n, float *a, float *r)
{
    sdata[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sdata[threadIdx.x] += a[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; i++)
             sdata[0] += sdata[i];
        r[0] = sdata[0];
    }
}

/**
 * Array power element wise.
 * @param n the number of elements.
 * @param a the array (output buffer).
 * @param s the power.
 * @return nothing.
 */
extern "C"
__global__ void pow_array(int n, float *a, int s)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        a[i] = pow(a[i], s);
    }
}

/**
 * Array subtraction element wise.
 * @param n the number of elements.
 * @param a1 the first array (output buffer).
 * @param a2 the second array.
 * @return nothing.
 */
extern "C"
__global__ void sub_array(int n, float *a1, float *a2)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        a1[i] -= a2[i];
    }
}

/**
 * Array multiplication element wise.
 * @param n the number of elements.
 * @param a1 the first array (output buffer).
 * @param a2 the second array.
 * @return nothing.
 */
extern "C"
__global__ void mul_array(int n, float *a1, float *a2)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        a1[i] *= a2[i];
    }
}

/**
 * Multiplication element wise with a scalar.
 * @param n the number of elements.
 * @param a the array (output buffer).
 * @param s the scalar.
 * @return nothing.
 */
extern "C"
__global__ void mul_float(int n, float *a, float s)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        a[i] *= s;
    }
}

/**
 * Array addition element wise.
 * @param n the number of elements.
 * @param a1 the first array (output buffer).
 * @param a2 the second array.
 * @return nothing.
 */
extern "C"
__global__ void add_array(int n, float *a1, float *a2)
{
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        a1[i] += a2[i];
    }
}
