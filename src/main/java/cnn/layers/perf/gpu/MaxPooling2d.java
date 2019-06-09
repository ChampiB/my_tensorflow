package cnn.layers.perf.gpu;

import cnn.layers.perf.MaxPooling2dInterface;
import cnn.layers.perf.gpu.useful.GPUMemoryHelper;
import cnn.layers.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.*;

public class MaxPooling2d extends GPUTask implements MaxPooling2dInterface {

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        super("max_pooling_2d.cu", new String[]{"training_activation", "activation", "inputs_gradients"});
    }

    /**
     * Compute the output shape.
     * @param kernel the kernel size.
     * @param x the input.
     * @return the shape.
     */
    private long[] computeOutputShape(int[] kernel, INDArray x) {
        return new long[]{
                x.shape()[0],
                x.shape()[1],
                x.shape()[2] / kernel[0],
                x.shape()[3] / kernel[1]
        };
    }

    /**
     * Compute the output size.
     * @param yShape the output shape.
     * @return the size.
     */
    private int computeOutputSize(long[] yShape) {
        return (int)(yShape[0] *  yShape[1] *  yShape[2] * yShape[3]);
    }

    /**
     * Create the configuration.
     * @param kernel the kernel size.
     * @param yShape the output shape.
     * @param xShape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(int[] kernel, long[] yShape, long[] xShape, int nbElements) {
        return new int[]{
                kernel[0], kernel[1],
                (int)(xShape[1] * xShape[2] * xShape[3]),
                (int)(xShape[2] * xShape[3]),
                (int)xShape[3],
                nbElements,
                (int)(yShape[1] * yShape[2] * yShape[3]),
                (int)(yShape[2] * yShape[3]),
                (int)(yShape[3])
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
        long[] yShape = computeOutputShape(kernel, x);
        int rsize = computeOutputSize(yShape);
        CUdeviceptr rgpu = GPUMemoryHelper.mallocFloatOutput(rsize);
        int msize = (int)x.length();
        CUdeviceptr mgpu = GPUMemoryHelper.mallocFloatOutput(msize);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(kernel, yShape, x.shape(), rsize));
        // Create kernel parameters.
        if (training) {
            Pointer parameters = Pointer.to(Pointer.to(cgpu), Pointer.to(xgpu), Pointer.to(rgpu), Pointer.to(mgpu));
            execute(
                    "training_activation", parameters,
                    (int)yShape[1], (int)yShape[2], (int)yShape[3],
                    (int)yShape[0], 1, 1,
                    rsize * Sizeof.FLOAT
            );
        } else {
            Pointer parameters = Pointer.to(Pointer.to(cgpu), Pointer.to(xgpu), Pointer.to(rgpu));
            execute(
                    "activation", parameters,
                    (int)yShape[1], (int)yShape[2], (int)yShape[3],
                    (int)yShape[0], 1, 1,
                    rsize * Sizeof.FLOAT
            );
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
        long[] yShape = g.shape();
        int size = (int)g.length();
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr ggpu = GPUMemoryHelper.mallocFloatInput(g);
        CUdeviceptr mgpu = GPUMemoryHelper.mallocFloatInput(m);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(kernel, g.shape(), m.shape(), size));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(ggpu),
                Pointer.to(mgpu)
        );
        execute(
                "inputs_gradients", parameters,
                (int)yShape[1], (int)yShape[2], (int)yShape[3],
                (int)yShape[0], 1, 1,
                size * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(mgpu, (int)m.length()).reshape(m.shape());
        // Clean up.
        cuMemFree(ggpu);
        cuMemFree(cgpu);
        cuMemFree(mgpu);
        return result;
    }
}
