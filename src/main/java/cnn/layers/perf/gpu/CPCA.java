package cnn.layers.perf.gpu;

import jcuda.Sizeof;
import cnn.layers.conf.ConfConv2d;
import cnn.layers.perf.CPCAInterface;
import cnn.layers.perf.gpu.useful.GPUMemoryHelper;
import cnn.layers.perf.gpu.useful.GPUTask;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

import static jcuda.driver.JCudaDriver.cuMemFree;

public class CPCA extends GPUTask implements CPCAInterface {

    /**
     * Default constructor.
     */
    public CPCA() {
        super("cpca.cu", new String[]{"weights_gradients", "count_patterns"});
    }

    /**
     * Compute the output size.
     * @param shape the weights's shape.
     * @return the output size.
     */
    private int computeOutputSize(long[] shape) {
        return (int)(shape[0] *  shape[1] *  shape[2] *  shape[3]);
    }

    /**
     * Normalize the CPCA gradients.
     */
    private INDArray computeMeanOfCPCA(INDArray dw, INDArray fc) {
        for (int fi = 0; fi < dw.shape()[0]; fi++) {
            double d = fc.getNumber(fi).doubleValue();
            if (d != 0) {
                dw.putSlice(fi, dw.slice(fi).div(d));
            }
        }
        return dw;
    }

    /**
     * Compute the weights gradients.
     * @param shape the weights's shape.
     * @param xgpu the inputs.
     * @param wgpu the weights.
     * @param ygpu the outputs.
     * @return the gradients.
     */
    private INDArray computeGradients(CUdeviceptr cgpu, int[] conf, long[] shape, CUdeviceptr xgpu, CUdeviceptr wgpu, CUdeviceptr ygpu) {
        // Allocate device output memory.
        int size = computeOutputSize(shape);
        CUdeviceptr rgpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(xgpu),
                Pointer.to(wgpu),
                Pointer.to(ygpu),
                Pointer.to(rgpu)
        );
        execute(
                "weights_gradients", kernelParameters,
                conf[8] * conf[9], conf[10], conf[11],
                5, 10, 10,
                500 * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray dw = GPUMemoryHelper.toCPU(rgpu, size).reshape(shape);
        // Clean up.
        cuMemFree(rgpu);
        return dw;
    }

    /**
     * Count the number of activated unit in each feature map.
     * @param size the number of feature map.
     * @param ygpu the layer's output.
     * @return the count.
     */
    private INDArray countPatterns(CUdeviceptr cgpu, int[] conf, int size, CUdeviceptr ygpu) {
        // Allocate device output memory.
        CUdeviceptr rgpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(ygpu),
                Pointer.to(rgpu)
        );
        execute(
                "count_patterns", kernelParameters,
                conf[3], 1, 1,
                conf[2], 5, 5,
                conf[2] * 25 * Sizeof.FLOAT
        );
        // Allocate host output memory and copy the device output to the host.
        INDArray fc = GPUMemoryHelper.toCPU(rgpu, size);
        // Clean up.
        cuMemFree(rgpu);
        return fc;
    }

    /**
     * Create the configuration.
     * @param xShape the input shape.
     * @param wShape the weights shape.
     * @param yShape the output activation.
     * @return the conf.
     */
    private int[] createConf(ConfConv2d conf, long[] xShape, long[] wShape, long[] yShape) {
        return new int[]{
                (int)(yShape[1] * yShape[2] * yShape[3]),
                (int)(yShape[2] * yShape[3]),
                (int)yShape[0], (int)yShape[1], (int)yShape[2], (int)yShape[3],
                conf.strides()[0], conf.strides()[1],
                (int)wShape[0], (int)wShape[1], (int)wShape[2], (int)wShape[3],
                (int)(xShape[1] * xShape[2] * xShape[3]),
                (int)(xShape[2] * xShape[3]),
                (int)(xShape[3]),
        };
    }

    /**
     * Compute the CPCA gradients.
     * @param conf the layer's configuration.
     * @param x the input.
     * @param w the weights.
     * @param y the layer output.
     * @return the gradients.
     */
    public INDArray weightsGradients(ConfConv2d conf, INDArray x, INDArray w, INDArray y) {
        // Create the kernel's configuration.
        int[] config = createConf(conf, x.shape(), w.shape(), y.shape());
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocFloatInput(w);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatInput(y);
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(config);
        // Execute kernels.
        INDArray dw = computeGradients(cgpu, config, w.shape(), xgpu, wgpu, ygpu);
        INDArray fc = countPatterns(cgpu, config, (int)w.shape()[0], ygpu);
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(wgpu);
        cuMemFree(ygpu);
        cuMemFree(cgpu);
        return computeMeanOfCPCA(dw, fc);
    }
}
