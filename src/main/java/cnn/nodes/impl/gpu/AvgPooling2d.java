package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.conf.Pooling2dConf;
import cnn.data.ArrayPtr;
import cnn.useful.gpu.GPUNode;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.security.InvalidParameterException;

public class AvgPooling2d extends GPUNode {

    private ArrayPtr cgpu = ArrayPtrFactory.empty();
    private ArrayPtr yagpu = ArrayPtrFactory.empty();
    private ArrayPtr yigpu = ArrayPtrFactory.empty();

    private long[] xShape;
    private int[] kernel;

    /**
     * Default constructor.
     */
    public AvgPooling2d() {
        this(new Pooling2dConf());
    }

    /**
     * Default constructor.
     * @param kernel the size of the pooling kernel.
     */
    public AvgPooling2d(int[] kernel) {
        this(new Pooling2dConf(kernel));
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public AvgPooling2d(Pooling2dConf conf) {
        super(GPUNode.NODES_PATH, "avg_pooling_2d.cu", new String[]{"activation", "inputs_gradients"});
        this.kernel = conf.getKernel();
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public AvgPooling2d(Object conf) {
        this((Pooling2dConf) conf);
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

    private ArrayPtr activation(ArrayPtr x) {
        // Allocate the output and configuration on device memory.
        xShape = x.getShape().clone();
        long[] yShape = computeOutputShape(kernel, x.getShape());
        int rsize = computeOutputSize(yShape);
        if (yagpu.isNull())
            yagpu = ArrayPtrFactory.empty(yShape, Sizeof.FLOAT);
        cgpu.copy(createConf(kernel, yShape, x.getShape(), rsize));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), yagpu.toPTR());
        execute(
                "activation", parameters,
                (int)yShape[1], (int)yShape[2], (int)yShape[3],
                (int)yShape[0], 1, 1,
                0
        );
        return yagpu;
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        return activation(x[0]);
    }

    private ArrayPtr update(ArrayPtr gradient) {
        // Allocate the device input data, and copy the host input data to the device.
        long[] yShape = gradient.getShape();
        if (yigpu.isNull())
            yigpu = ArrayPtrFactory.empty(xShape, Sizeof.FLOAT);
        cgpu.copy(createConf(kernel, yShape, xShape, gradient.getSize()));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), gradient.toPTR(), yigpu.toPTR());
        execute(
                "inputs_gradients", parameters,
                (int)yShape[1], (int)yShape[2], (int)yShape[3],
                (int)yShape[0], 1, 1,
                0
        );
        return yigpu;
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        if (gradient.length != 1)
            throw new InvalidParameterException();
        return new ArrayPtr[]{update(gradient[0])};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "AvgPooling2d");
        kryo.writeObject(output, kernel);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: AvgPooling2d(gpu)");
    }
}
