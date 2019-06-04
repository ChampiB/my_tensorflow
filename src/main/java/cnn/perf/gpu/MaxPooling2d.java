package cnn.perf.gpu;

import cnn.perf.MaxPooling2dInterface;
import cnn.perf.gpu.useful.GPUMemoryHelper;
import cnn.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class MaxPooling2d extends GPUTask implements MaxPooling2dInterface {

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        super("max_pooling_2d.cu", new String[]{"training_max_pooling_2d", "max_pooling_2d", "inputs_gradients"});
    }

    /**
     * Compute the output shape.
     * @param kernel the kernel size.
     * @param x the input.
     * @return the shape.
     */
    private int[] computeOutputShape(int[] kernel, INDArray x) {
        return new int[]{
                (int) x.shape()[0],
                (int) x.shape()[1],
                (int) x.shape()[2] / kernel[0],
                (int) x.shape()[3] / kernel[1]
        };
    }

    /**
     * Compute the output size.
     * @param kernel the kernel size.
     * @param x the input.
     * @return the size.
     */
    private int computeOutputSize(int[] kernel, INDArray x) {
        int[] shape = computeOutputShape(kernel, x);
        return shape[0] *  shape[1] *  shape[2] * shape[3];
    }

    /**
     * Create the configuration.
     * @param kernel the kernel size.
     * @param shape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(int[] kernel, long[] shape, int nbElements) {
        return new int[]{
                kernel[0], kernel[1], (int)shape[1], (int)shape[2], (int)shape[3], nbElements
        };
    }

    /**
     * Compute max pooling.
     * @param kernel the kernel/pooling size.
     * @param x the input.
     * @param training true if training and false otherwise.
     * @return a pair containing the output and the mask if training is true.
     */
    public Pair<INDArray, INDArray> maxPooling2d(int[] kernel, INDArray x, boolean training) {
        // Allocate device output memory.
        int rsize = computeOutputSize(kernel, x);
        CUdeviceptr rgpu = GPUMemoryHelper.mallocFloatOutput(rsize);
        int msize = (int)x.length();
        CUdeviceptr mgpu = GPUMemoryHelper.mallocFloatOutput(msize);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(kernel, x.shape(), rsize));
        // Create kernel parameters.
        if (training) {
            Pointer parameters = Pointer.to(Pointer.to(cgpu), Pointer.to(xgpu), Pointer.to(rgpu), Pointer.to(mgpu));
            execute("training_max_pooling_2d", parameters, rsize);
        } else {
            Pointer parameters = Pointer.to(Pointer.to(cgpu), Pointer.to(xgpu), Pointer.to(rgpu));
            execute("max_pooling_2d", parameters, rsize);
        }
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(rgpu, rsize).reshape(computeOutputShape(kernel, x));
        INDArray mask = GPUMemoryHelper.toCPU(mgpu, msize).reshape(x.shape());
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(rgpu);
        cuMemFree(mgpu);
        cuMemFree(cgpu);
        return new ImmutablePair<>(result, mask);
    }

    /**
     * Compute the gradient with respect to the inputs.
     * @param kernel the kernel/pooling size.
     * @param g the gradient with respect to the output.
     * @param m the pooling mask.
     * @return the gradients.
     */
    public INDArray inputsGradients(int[] kernel, INDArray g, INDArray m) {
        // Compute the output size.
        int size = (int)g.length();
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr ggpu = GPUMemoryHelper.mallocFloatInput(g);
        CUdeviceptr mgpu = GPUMemoryHelper.mallocFloatInput(m);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(kernel, m.shape(), size));
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(ggpu),
                Pointer.to(mgpu)
        );
        execute("inputs_gradients", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(mgpu, (int)m.length()).reshape(m.shape());
        // Clean up.
        cuMemFree(ggpu);
        cuMemFree(cgpu);
        cuMemFree(mgpu);
        return result;
    }
}
