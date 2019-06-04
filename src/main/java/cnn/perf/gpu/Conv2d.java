package cnn.perf.gpu;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.Conv2dInterface;
import cnn.perf.gpu.useful.GPUTask;
import cnn.perf.gpu.useful.GPUMemoryHelper;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class Conv2d extends GPUTask implements Conv2dInterface {

    /**
     * Default constructor.
     */
    public Conv2d() {
        super("conv_2d.cu", new String[]{"conv_2d", "inputs_gradients", "weights_gradients"});
    }

    /**
     * Compute the output shape.
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output shape.
     */
    public int[] computeOutputShape(ConfConv2d conf, INDArray x) {
        // Compute the number of vertical and horizontal position.
        long nr = x.shape()[2] - conf.filters()[1] + 1;
        nr = (long) Math.ceil(((double)nr) / ((double)conf.strides()[0]));
        long nc = x.shape()[3] - conf.filters()[2] + 1;
        nc = (long) Math.ceil(((double)nc) / ((double)conf.strides()[1]));
        // Format the output.
        return new int[]{(int) x.shape()[0], conf.filters()[0], (int) nr, (int) nc};
    }

    /**
     * Compute the output size (conv2d).
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output size.
     */
    public int computeOutputSize(ConfConv2d conf, INDArray x) {
        int[] shape = computeOutputShape(conf, x);
        return shape[0] * shape[1] * shape[2] * shape[3];
    }

    /**
     * Compute the output size (gradients).
     * @param shape the output shape.
     * @return the output size.
     */
    public int computeOutputSize(long[] shape) {
        return (int)(shape[0] * shape[1] * shape[2] * shape[3]);
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param x the input.
     * @param w the weights.
     * @param bw the bias weights.
     * @return the output.
     */
    public INDArray conv2d(ConfConv2d conf, INDArray x, INDArray w, INDArray bw) {
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocInput(x);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocInput(w);
        CUdeviceptr bwgpu = GPUMemoryHelper.mallocInput(bw);
        // Allocate device output memory.
        int size = computeOutputSize(conf, x);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocOutput(size);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{
                        conf.filters()[0],
                        conf.filters()[1],
                        conf.filters()[2],
                        conf.strides()[0],
                        conf.strides()[1]
                }),
                Pointer.to(xgpu),
                Pointer.to(wgpu),
                Pointer.to(bwgpu),
                Pointer.to(ygpu)
        );
        execute("conv_2d", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu).reshape(computeOutputShape(conf, x));
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(wgpu);
        cuMemFree(bwgpu);
        cuMemFree(ygpu);
        return result;
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray inputsGradients(ConfConv2d conf, long[] yShape, INDArray w, INDArray g) {
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr ggpu = GPUMemoryHelper.mallocInput(g);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocInput(w);
        // Allocate device output memory.
        int size = computeOutputSize(yShape);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocOutput(size);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{
                        conf.filters()[0],
                        conf.filters()[1],
                        conf.filters()[2],
                        conf.strides()[0],
                        conf.strides()[1]
                }),
                Pointer.to(wgpu),
                Pointer.to(ggpu),
                Pointer.to(ygpu)
        );
        execute("inputs_gradient", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu).reshape(yShape);
        // Clean up.
        cuMemFree(ggpu);
        cuMemFree(wgpu);
        cuMemFree(ygpu);
        return result;
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray weightsGradients(ConfConv2d conf, long[] yShape, INDArray x, INDArray g) {
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr ggpu = GPUMemoryHelper.mallocInput(g);
        CUdeviceptr xgpu = GPUMemoryHelper.mallocInput(x);
        // Allocate device output memory.
        int size = computeOutputSize(yShape);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocOutput(size);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{
                        conf.filters()[0],
                        conf.filters()[1],
                        conf.filters()[2],
                        conf.strides()[0],
                        conf.strides()[1]
                }),
                Pointer.to(xgpu),
                Pointer.to(ggpu),
                Pointer.to(ygpu)
        );
        execute("weights_gradient", kernelParameters, size);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu).reshape(yShape);
        // Clean up.
        cuMemFree(ggpu);
        cuMemFree(xgpu);
        cuMemFree(ygpu);
        return result;
    }
}
