package cnn.perf.gpu;

import jcuda.Sizeof;
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
    private INDArray computeGradients(int[] conf, long[] shape, CUdeviceptr xgpu, CUdeviceptr wgpu, CUdeviceptr ygpu) {
        // Allocate device output memory.
        int size = computeOutputSize(shape);
        CUdeviceptr rgpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(conf);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(xgpu),
                Pointer.to(wgpu),
                Pointer.to(ygpu),
                Pointer.to(rgpu)
        );
        execute("weights_gradients", kernelParameters, size, 256, 256 * Sizeof.FLOAT);
        // Allocate host output memory and copy the device output to the host.
        INDArray dw = GPUMemoryHelper.toCPU(rgpu, size).reshape(shape);
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(rgpu);
        return dw;
    }

    /**
     * Count the number of activated unit in each feature map.
     * @param size the number of feature map.
     * @param ygpu the layer's output.
     * @return the count.
     */
    private INDArray countPatterns(int[] conf, int size, CUdeviceptr ygpu) {
        // Allocate device output memory.
        CUdeviceptr rgpu = GPUMemoryHelper.mallocFloatOutput(size);
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr cgpu = GPUMemoryHelper.mallocIntInput(conf);
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(cgpu),
                Pointer.to(ygpu),
                Pointer.to(rgpu)
        );
        execute("count_patterns", kernelParameters, conf[1], 256, 256 * Sizeof.FLOAT);
        // Allocate host output memory and copy the device output to the host.
        INDArray fc = GPUMemoryHelper.toCPU(rgpu, size);
        // Clean up.
        cuMemFree(cgpu);
        cuMemFree(rgpu);
        return fc;
    }

    /**
     * Create the configuration.
     * @param x the input activation.
     * @param w the weights.
     * @param y the output activation.
     * @return the conf.
     */
    private int[] createConf(ConfConv2d conf, INDArray x, INDArray w, INDArray y) {
        return new int[]{
                (int)y.shape()[0], (int)y.shape()[1], (int)y.shape()[2], (int)y.shape()[3],
                (int)w.shape()[1], (int)w.shape()[2], (int)w.shape()[3],
                (int)x.shape()[0], (int)x.shape()[1], (int)x.shape()[2], (int)x.shape()[3],
                conf.strides()[0], conf.strides()[1]
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
        // Allocate the device input data, and copy the host input data to the device.
        CUdeviceptr xgpu = GPUMemoryHelper.mallocFloatInput(x);
        CUdeviceptr wgpu = GPUMemoryHelper.mallocFloatInput(w);
        CUdeviceptr ygpu = GPUMemoryHelper.mallocFloatInput(y);
        // Execute kernels.
        INDArray dw = computeGradients(createConf(conf, x, w, y), w.shape(), xgpu, wgpu, ygpu);
        INDArray fc = countPatterns(createConf(conf, x, w, y), (int)w.shape()[0], ygpu);
        // Clean up.
        cuMemFree(xgpu);
        cuMemFree(wgpu);
        cuMemFree(ygpu);
        return computeMeanOfCPCA(dw, fc);
    }
}
