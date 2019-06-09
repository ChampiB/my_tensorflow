package cnn.layers.perf.gpu;

import cnn.layers.conf.ConfConv2d;
import cnn.layers.perf.KWTAInterface;
import cnn.layers.perf.gpu.useful.GPUMemoryHelper;
import cnn.layers.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class KWTA extends GPUTask implements KWTAInterface {

    /**
     * Default constructor.
     */
    public KWTA() {
        super("kwta.cu", new String[]{"activation"});
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
     * Create the configuration.
     * @param shape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(long[] shape, int nbElements) {
        return new int[]{
                (int)(shape[1] * shape[2] * shape[3]), (int)(shape[2] * shape[3]), (int)shape[3], nbElements
        };
    }

    /**
     * Compute the k-winners-take-all activation.
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output.
     */
    public INDArray activation(ConfConv2d conf, INDArray x) {
        // Allocate device output memory.
        int size = computeOutputSize(x);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(x.shape(), size));
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(new int[]{conf.k()}),
                Pointer.to(xgpu),
                Pointer.to(ygpu)
        );
        execute(
                "activation", kernelParameters,
                (int)x.shape()[2], (int)x.shape()[3], 1,
                (int)x.shape()[0], 1, 1,
                size * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(x.shape());
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(cgpu);
        cuMemFree(ygpu);
        return result;
    }
}
