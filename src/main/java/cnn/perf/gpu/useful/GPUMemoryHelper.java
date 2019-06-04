package cnn.perf.gpu.useful;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static jcuda.driver.JCudaDriver.*;

public class GPUMemoryHelper {

    /**
     * Allocate the device input data, and copy the host input data to the device.
     * @param x the input array.
     * @return the gpu pointer.
     */
    static public CUdeviceptr mallocInput(INDArray x) {
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
    static public CUdeviceptr mallocOutput(int size) {
        CUdeviceptr output = new CUdeviceptr();
        cuMemAlloc(output, size * Sizeof.FLOAT);
        return output;
    }

    /**
     * Transfer the data from the GPU to the CPU.
     * @param ygpu the GPU pointer.
     * @return the INDArray (CPU pointer).
     */
    static public INDArray toCPU(CUdeviceptr ygpu) {
        int size = ygpu.getByteBuffer().array().length;
        float[] buffer = new float[size];
        cuMemcpyDtoH(Pointer.to(buffer), ygpu, size * Sizeof.FLOAT);
        return Nd4j.create(buffer);
    }
}
