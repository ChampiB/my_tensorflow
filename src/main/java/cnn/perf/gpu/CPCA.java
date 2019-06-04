package cnn.perf.gpu;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.CPCAInterface;
import cnn.perf.gpu.useful.GPUMemoryHelper;
import cnn.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class CPCA extends GPUTask implements CPCAInterface {

    /**
     * Default constructor.
     */
    public CPCA() {
        super("cpca.cu", new String[]{"weights_gradients"});
    }

    /**
     * Compute the output size.
     * @param w the weights.
     * @return the output size.
     */
    public int computeOutputSize(INDArray w) {
        return (int)(w.shape()[0] *  w.shape()[1] *  w.shape()[2] *  w.shape()[3]);
    }

    /**
     * Compute the CPCA gradients.
     * @param x the input.
     * @param w the weights.
     * @param y the layer output.
     * @return the gradients.
     */
    public INDArray weightsGradients(ConfConv2d conf, INDArray x, INDArray w, INDArray y) {
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocInput(x);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocInput(w);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocInput(y);
        // Allocate device output memory.
        int size = computeOutputSize(w);
        CUdeviceptr rgpu = GPUMemoryHelper.mallocOutput(size);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(xgpu),
                Pointer.to(wgpu),
                Pointer.to(ygpu),
                Pointer.to(rgpu)
        );
        execute("weights_gradients", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(rgpu).reshape(w.shape());
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(wgpu);
        cuMemFree(ygpu);
        cuMemFree(rgpu);
        return result;
    }
}
