package cnn.layers.perf.gpu;

import cnn.layers.conf.ConfConv2d;
import cnn.layers.perf.Conv2dInterface;
import cnn.layers.perf.gpu.useful.GPUTask;
import cnn.layers.perf.gpu.useful.GPUMemoryHelper;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class Conv2d extends GPUTask implements Conv2dInterface {

    /**
     * Default constructor.
     */
    public Conv2d() {
        super("conv_2d.cu", new String[]{"activation", "inputs_gradients", "weights_gradients"});
    }

    /**
     * Compute the output shape.
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output shape.
     */
    private long[] computeOutputShape(ConfConv2d conf, INDArray x) {
        // Compute the number of vertical and horizontal position.
        long nr = x.shape()[2] - conf.filters()[1] + 1;
        nr = (long) Math.ceil(((double)nr) / ((double)conf.strides()[0]));
        long nc = x.shape()[3] - conf.filters()[2] + 1;
        nc = (long) Math.ceil(((double)nc) / ((double)conf.strides()[1]));
        // Format the output.
        return new long[]{x.shape()[0], conf.filters()[0], nr, nc};
    }

    /**
     * Compute the output size (activation).
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output size.
     */
    private int computeOutputSize(ConfConv2d conf, INDArray x) {
        long[] shape = computeOutputShape(conf, x);
        return (int)(shape[0] * shape[1] * shape[2] * shape[3]);
    }

    /**
     * Compute the output size (gradients).
     * @param shape the output shape.
     * @return the output size.
     */
    private int computeOutputSize(long[] shape) {
        return (int)(shape[0] * shape[1] * shape[2] * shape[3]);
    }

    /**
     * Create the configuration.
     * @param conf the layer configuration.
     * @param wshape the weights shape.
     * @param xshape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(ConfConv2d conf, long[] wshape, long[] xshape, int nbElements, long[] yshape) {
        return new int[]{
                conf.filters()[0], conf.filters()[1], conf.filters()[2],
                conf.strides()[0], conf.strides()[1],
                (int)xshape[1], (int)xshape[2], (int)xshape[3],
                nbElements,
                (int)yshape[0], (int)yshape[1], (int)yshape[2], (int)yshape[3],
                (int)wshape[0], (int)wshape[1], (int)wshape[2], (int)wshape[3],
                (int)(yshape[1] * yshape[2] * yshape[3]),
                (int)(yshape[2] * yshape[3]),
                (int)(yshape[3]),
                (int)(xshape[1] * xshape[2] * xshape[3]),
                (int)(xshape[2] * xshape[3]),
                (int)(xshape[3])
        };
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param x the input.
     * @param w the weights.
     * @param bw the bias weights.
     * @return the output.
     */
    public INDArray activation(ConfConv2d conf, INDArray x, INDArray w, INDArray bw) {
        // Allocate device output memory.
        long[] shape = computeOutputShape(conf, x);
        int size = computeOutputSize(conf, x);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(conf, w.shape(), x.shape(), size, shape));
        CUdeviceptr wgpu = GPUMemoryHelper.mallocFloatInput(w);
        CUdeviceptr bwgpu = GPUMemoryHelper.mallocFloatInput(bw);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu), Pointer.to(xgpu), Pointer.to(wgpu), Pointer.to(bwgpu), Pointer.to(ygpu)
        );
        execute(
                "activation", kernelParameters,
                (int)shape[1], (int)shape[2], (int)shape[3],
                (int)shape[0], 1, 1,
                size * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(shape);
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(cgpu);
        cuMemFree(wgpu);
        cuMemFree(bwgpu);
        cuMemFree(ygpu);
        return result;
    }

    /**
     * Compute the gradients with respect to the inputs.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray inputsGradients(ConfConv2d conf, long[] yShape, INDArray w, INDArray g) {
        // Allocate device output memory.
        int size = computeOutputSize(yShape);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(createConf(conf, w.shape(), yShape, size, g.shape()));
        CUdeviceptr ggpu = GPUMemoryHelper.mallocFloatInput(g);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocFloatInput(w);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu), Pointer.to(wgpu), Pointer.to(ggpu), Pointer.to(ygpu)
        );
        execute("inputs_gradients", kernelParameters, size, 256, 256 * Sizeof.FLOAT);
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(yShape);
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(ggpu);
        cuMemFree(wgpu);
        cuMemFree(ygpu);
        return result;
    }

    /**
     * Compute the gradient with respect to the weights.
     * @param conf the layer configuration.
     * @param dwShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the weights.
     */
    public INDArray weightsGradients(ConfConv2d conf, long[] dwShape, INDArray x, INDArray g) {
        // Allocate device output memory.
        int size = computeOutputSize(dwShape);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        int[] config = createConf(conf, dwShape, x.shape(), size, g.shape());
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(config);
        CUdeviceptr ggpu = GPUMemoryHelper.mallocFloatInput(g);
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu), Pointer.to(xgpu), Pointer.to(ggpu), Pointer.to(ygpu)
        );
        execute(
                "weights_gradients", kernelParameters,
                config[13] * config[14], config[15], config[16],
                5, 10, 10,
                500 * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray result = GPUMemoryHelper.toCPU(ygpu, size).reshape(dwShape);
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(ggpu);
        cuMemFree(xgpu);
        cuMemFree(ygpu);
        return result;
    }
}
