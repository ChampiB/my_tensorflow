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
        super("cpca.cu", new String[]{"max_pooling_2d", "inputs_gradients"});
    }

    /**
     * Compute the output shape.
     * @param kernel the kernel size.
     * @param x the input.
     * @return the shape.
     */
    public int[] computeOutputShape(int[] kernel, INDArray x) {
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
    public int computeOutputSize(int[] kernel, INDArray x) {
        int[] shape = computeOutputShape(kernel, x);
        return shape[0] *  shape[1] *  shape[2] * shape[3];
    }

    /**
     * Compute max pooling.
     * @param kernel the kernel/pooling size.
     * @param x the input.
     * @param training true if training and false otherwise.
     * @return a pair containing the output and the mask if training is true.
     */
    public Pair<INDArray, INDArray> maxPooling2d(int[] kernel, INDArray x, boolean training) {
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocInput(x);
        // Allocate device output memory.
        int rsize = computeOutputSize(kernel, x);
        CUdeviceptr rgpu = GPUMemoryHelper.mallocOutput(rsize);
        int msize = (int)x.length();
        CUdeviceptr mgpu = GPUMemoryHelper.mallocOutput(msize);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(kernel),
                Pointer.to(xgpu),
                Pointer.to(rgpu),
                Pointer.to(mgpu)
        );
        execute("max_pooling_2d", kernelParameters, msize);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(rgpu).reshape(computeOutputShape(kernel, x));
        INDArray mask = GPUMemoryHelper.toCPU(mgpu).reshape(x.shape());
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(rgpu);
        cuMemFree(mgpu);
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
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr ggpu = GPUMemoryHelper.mallocInput(g);
        CUdeviceptr mgpu = GPUMemoryHelper.mallocInput(m);
        // Allocate device output memory.
        int size = (int)g.length();
        CUdeviceptr rgpu = GPUMemoryHelper.mallocOutput(size);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(kernel),
                Pointer.to(ggpu),
                Pointer.to(mgpu),
                Pointer.to(rgpu)
        );
        execute("inputs_gradients", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(rgpu).reshape(g.shape());
        // Clean up.
        cuMemFree(ggpu);
        cuMemFree(mgpu);
        cuMemFree(rgpu);
        return result;
    }
}
