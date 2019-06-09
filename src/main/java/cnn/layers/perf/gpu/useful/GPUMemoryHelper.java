package cnn.layers.perf.gpu.useful;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static jcuda.driver.JCudaDriver.*;

public class GPUMemoryHelper {

    /**
     * Allocate the device input data, and copy the host input data to the device.
     * @param data the input array.
     * @return the gpu pointer.
     */
    static public CUdeviceptr mallocIntInput(int[] data) {
        CUdeviceptr input = new CUdeviceptr();
        cuMemAlloc(input, data.length * Sizeof.INT);
        cuMemcpyHtoD(input, Pointer.to(data), data.length * Sizeof.INT);
        return input;
    }


    /**
     * Allocate the device input data, and copy the host input data to the device.
     * @param x the input array.
     * @return the gpu pointer.
     */
    static public CUdeviceptr mallocFloatInput(INDArray x) {
        float[] data = x.data().asFloat();
        CUdeviceptr input = new CUdeviceptr();
        cuMemAlloc(input, data.length * Sizeof.FLOAT);
        cuMemcpyHtoD(input, Pointer.to(data), data.length * Sizeof.FLOAT);
        return input;
    }

    /**
     * Allocate the device output data.
     * @param size the number of elements in the output buffer.
     * @return the gpu pointer.
     */
    static public CUdeviceptr mallocFloatOutput(int size) {
        CUdeviceptr output = new CUdeviceptr();
        cuMemAlloc(output, size * Sizeof.FLOAT);
        return output;
    }

    /**
     * Transfer the data from the GPU to the CPU.
     * @param a the GPU pointer.
     * @return the INDArray (CPU pointer).
     */
    static public INDArray toCPU(CUdeviceptr a, int size) {
        float[] buffer = new float[size];
        cuMemcpyDtoH(Pointer.to(buffer), a, size * Sizeof.FLOAT);
        return Nd4j.create(buffer);
    }
}
