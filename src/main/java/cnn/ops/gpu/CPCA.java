package cnn.ops.gpu;

import cnn.useful.ArrayPtr;
import jcuda.Sizeof;
import cnn.layers.conf.Conv2dConf;
import cnn.ops.CPCAInterface;
import cnn.useful.gpu.GPUTask;
import jcuda.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CPCA extends GPUTask implements CPCAInterface {

    /**
     * Default constructor.
     */
    public CPCA() {
        super(GPUTask.OPS_PATH, "cpca.cu", new String[]{"weights_gradients", "count_patterns"});
    }

    /**
     * Normalize the CPCA gradients.
     */
    private INDArray computeMeanOfCPCA(ArrayPtr dw, ArrayPtr fc) {
        for (int fi = 0; fi < dw.getShape()[0]; fi++) {
            double d = fc.toCPU().getNumber(fi).doubleValue();
            if (d != 0) {
                dw.toCPU().putSlice(fi, dw.toCPU().slice(fi).div(d));
            }
        }
        return dw.toCPU();
    }

    /**
     * Compute the weights gradients.
     * @param shape the weights's shape.
     * @param x the inputs.
     * @param w the weights.
     * @param y the outputs.
     * @return the gradients.
     */
    private ArrayPtr computeGradients(ArrayPtr c, int[] conf, long[] shape, ArrayPtr x, ArrayPtr w, ArrayPtr y) {
        // Allocate device output memory.
        ArrayPtr rgpu = new ArrayPtr(shape, Sizeof.FLOAT);
        // Create kernel parameters.
        Pointer parameters = Pointer.to(c.toPTR(), x.toPTR(), w.toPTR(), y.toPTR(), rgpu.toPTR());
        execute(
                "weights_gradients", parameters,
                conf[8] * conf[9], conf[10], conf[11],
                5, 10, 10,
                500 * Sizeof.FLOAT
        );
        return rgpu;
    }

    /**
     * Count the number of activated unit in each feature map.
     * @param size the number of feature map.
     * @param ygpu the layer's output.
     * @return the count.
     */
    private ArrayPtr countPatterns(ArrayPtr cgpu, int[] conf, int size, ArrayPtr ygpu) {
        // Allocate device output memory.
        ArrayPtr rgpu = new ArrayPtr(new long[]{size}, Sizeof.FLOAT);
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), ygpu.toPTR(), rgpu.toPTR());
        execute(
                "count_patterns", parameters,
                conf[3], 1, 1,
                conf[2], 5, 5,
                conf[2] * 25 * Sizeof.FLOAT
        );
        return rgpu;
    }

    /**
     * Create the configuration.
     * @param xShape the input shape.
     * @param wShape the weights shape.
     * @param yShape the output activation.
     * @return the conf.
     */
    private int[] createConf(Conv2dConf conf, long[] xShape, long[] wShape, long[] yShape) {
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
    public INDArray weightsGradients(Conv2dConf conf, ArrayPtr x, ArrayPtr w, ArrayPtr y) {
        // Create the kernel's configuration.
        int[] config = createConf(conf, x.getShape(), w.getShape(), y.getShape());
        // Allocate the device input data, and copy the host input data to the device.
        ArrayPtr cgpu = new ArrayPtr(config, true);
        // Execute kernels.
        ArrayPtr dw = computeGradients(cgpu, config, w.getShape(), x, w, y);
        ArrayPtr fc = countPatterns(cgpu, config, (int)w.getShape()[0], y);
        INDArray res = computeMeanOfCPCA(dw, fc);
        // Clean up.
        dw.freeGpu();
        fc.freeGpu();
        cgpu.freeGpu();
        return res;
    }
}
