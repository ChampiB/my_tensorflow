package cnn.perf.gpu;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.KWTAInterface;
import cnn.perf.gpu.useful.GPUMemoryHelper;
import cnn.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class KWTA extends GPUTask implements KWTAInterface {

    /**
     * Default constructor.
     */
    public KWTA() {
        super("kwta.cu", new String[]{"kwta"});
    }

    /**
     * Compute the output size.
     * @param x the input.
     * @return the output size.
     */
    public int computeOutputSize(INDArray x) {
        return (int)(x.shape()[0] * x.shape()[1] * x.shape()[2] * x.shape()[3]);
    }

    /**
     * Compute the k-winners-take-all activation.
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output.
     */
    public INDArray kwta(ConfConv2d conf, INDArray x) {
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocInput(x);
        // Allocate device output memory.
        int size = computeOutputSize(x);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocOutput(size);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{conf.k()}),
                Pointer.to(xgpu),
                Pointer.to(ygpu)
        );
        execute("kwta", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu).reshape(x.shape());
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(ygpu);
        return result;
    }
}
