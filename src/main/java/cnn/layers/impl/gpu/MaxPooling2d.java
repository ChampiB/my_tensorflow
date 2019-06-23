package cnn.layers.impl.gpu;

import cnn.layers.Layer;
import cnn.layers.conf.MaxPooling2dConf;
import cnn.useful.ArrayPtr;
import cnn.useful.gpu.GPUTask;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;

public class MaxPooling2d extends GPUTask implements Layer {

    private ArrayPtr mgpu = new ArrayPtr();
    private ArrayPtr cgpu = new ArrayPtr();
    private ArrayPtr yagpu = new ArrayPtr();

    private int[] kernel;

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        this(new MaxPooling2dConf());
    }

    /**
     * Default constructor.
     * @param kernel the size of the pooling kernel.
     */
    public MaxPooling2d(int[] kernel) {
        this(new MaxPooling2dConf(kernel));
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public MaxPooling2d(MaxPooling2dConf conf) {
        super(GPUTask.LAYERS_PATH, "max_pooling_2d.cu", new String[]{"training_activation", "activation", "inputs_gradients"});
        this.kernel = conf.getKernel();
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public MaxPooling2d(Object conf) {
        this((MaxPooling2dConf) conf);
    }

    /**
     * Compute the output shape.
     * @param kernel the kernel size.
     * @param shape the input' shape.
     * @return the shape.
     */
    private long[] computeOutputShape(int[] kernel, long[] shape) {
        return new long[]{
                shape[0],
                shape[1],
                shape[2] / kernel[0],
                shape[3] / kernel[1]
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

    @Override
    public ArrayPtr activation(ArrayPtr x, boolean training) {
        // Allocate the output and configuration on device memory.
        long[] yShape = computeOutputShape(kernel, x.getShape());
        int rsize = computeOutputSize(yShape);
        if (yagpu.isNull())
            yagpu = new ArrayPtr(yShape, Sizeof.FLOAT);
        cgpu.copy(createConf(kernel, yShape, x.getShape(), rsize));
        // Create kernel parameters.
        if (training) {
            if (mgpu.isNull())
                mgpu = new ArrayPtr(x.getShape(), Sizeof.FLOAT);
            Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), yagpu.toPTR(), mgpu.toPTR());
            execute(
                    "training_activation", parameters,
                    (int)yShape[1], (int)yShape[2], (int)yShape[3],
                    (int)yShape[0], 1, 1,
                    rsize * Sizeof.FLOAT
            );
        } else {
            Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), yagpu.toPTR());
            execute(
                    "activation", parameters,
                    (int)yShape[1], (int)yShape[2], (int)yShape[3],
                    (int)yShape[0], 1, 1,
                    rsize * Sizeof.FLOAT
            );
        }
        return yagpu;
    }

    @Override
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        // Compute the output size.
        long[] yShape = gradient.getShape();
        // Allocate the device input data, and copy the host input data to the device.
        cgpu.copy(createConf(kernel, yShape, mgpu.getShape(), gradient.getSize()));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), gradient.toPTR(), mgpu.toPTR());
        execute(
                "inputs_gradients", parameters,
                (int)yShape[1], (int)yShape[2], (int)yShape[3],
                (int)yShape[0], 1, 1,
                (int)yShape[0] * Sizeof.FLOAT
        );
        return mgpu;
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "MaxPooling2d");
        kryo.writeObject(output, kernel);
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public Layer load(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: MaxPooling2d(gpu)");
        System.out.println();
    }
}
