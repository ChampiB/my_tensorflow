package cnn.layers.perf.gpu;

import cnn.layers.perf.DenseInterface;
import cnn.layers.perf.gpu.useful.GPUMemoryHelper;
import cnn.layers.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class Dense extends GPUTask implements DenseInterface {

    /**
     * Default constructor.
     */
    public Dense() {
        super("dense.cu", new String[]{"activation", "inputs_gradients", "weights_gradients"});
    }

    /**
     * Crete the configuration.
     * @param nImage the number of images.
     * @param nInput the number of input neurons.
     * @param nOutput the number of output neurons.
     * @return the configuration.
     */
    private int[] createConf(long nImage, long nInput, long nOutput) {
        return new int[]{(int)nImage, (int)nInput, (int)nOutput};
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param x the input.
     * @param w the weights.
     * @return the output.
     */
    public INDArray activation(INDArray x, INDArray w) {
        // Allocate device output memory.
        int size = (int)(x.shape()[0] * w.shape()[1]);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(x.shape()[0], x.shape()[1], w.shape()[1]));
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocFloatInput(w);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(xgpu),
                Pointer.to(wgpu),
                Pointer.to(ygpu)
        );
        execute(
                "activation", kernelParameters,
                (int)x.shape()[0], (int)w.shape()[1], 1,
                512, 1, 1,
                512 * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(new long[]{x.shape()[0], w.shape()[1]});
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(xgpu);
        cuMemFree(wgpu);
        cuMemFree(ygpu);
        return result;
    }

    /**
     * Compute the gradients with respect to the inputs.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray inputsGradients(long[] yShape, INDArray w, INDArray g) {
        // Allocate device output memory.
        int size = (int)(yShape[0] * (yShape[1] - 1));
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(yShape[0], yShape[1], g.shape()[1]));
        CUdeviceptr wgpu = GPUMemoryHelper.mallocFloatInput(w);
        CUdeviceptr ggpu = GPUMemoryHelper.mallocFloatInput(g);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(wgpu),
                Pointer.to(ggpu),
                Pointer.to(ygpu)
        );
        execute(
                "inputs_gradients", kernelParameters,
                (int)yShape[0], (int)yShape[1] - 1, 1,
                32, 1, 1,
                32 * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(new long[]{yShape[0], yShape[1] - 1});
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(wgpu);
        cuMemFree(ggpu);
        cuMemFree(ygpu);
        return result;
    }


    /**
     * Compute the gradient with respect to the weights.
     * @param dwShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the weights.
     */
    public INDArray weightsGradients(long[] dwShape, INDArray x, INDArray g) {
        // Allocate device output memory.
        int size = (int)(dwShape[0] * dwShape[1]);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(x.shape()[0], x.shape()[1], g.shape()[1]));
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr ggpu = GPUMemoryHelper.mallocFloatInput(g);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(xgpu),
                Pointer.to(ggpu),
                Pointer.to(ygpu)
        );
        execute(
                "weights_gradients", kernelParameters,
                (int)g.shape()[1], (int)x.shape()[1], 1,
                32, 1, 1,
                32 * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(dwShape[0], dwShape[1]);
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(xgpu);
        cuMemFree(ggpu);
        cuMemFree(ygpu);
        return result;
    }
}
