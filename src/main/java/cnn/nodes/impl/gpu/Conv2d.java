package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.NodesFactory;
import cnn.nodes.conf.PadConf;
import cnn.ops.*;
import cnn.nodes.conf.Conv2dConf;
import cnn.data.ArrayPtr;
import cnn.useful.gpu.GPUNode;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.security.InvalidParameterException;
import java.util.Arrays;

import static cnn.nodes.enumerations.PaddingType.SAME;

/**
 * GPU implementation of the convolutional layer.
 */
public class Conv2d extends GPUNode {

    private ArrayPtr bwgpu = ArrayPtrFactory.empty();
    private ArrayPtr wgpu = ArrayPtrFactory.empty();
    private ArrayPtr cgpu = ArrayPtrFactory.empty();
    private ArrayPtr yagpu = ArrayPtrFactory.empty();
    private ArrayPtr yigpu = ArrayPtrFactory.empty();
    private ArrayPtr ywgpu = ArrayPtrFactory.empty();
    private ArrayPtr input = ArrayPtrFactory.empty();

    private long[] inputShape = null;
    private Conv2dConf conf;

    private CPCAInterface cpca = OpsFactory.create("CPCA", "gpu");
    private OperationInterface op = OpsFactory.create("Operation", "gpu");
    private Node kwta;
    private Node activation;

    /**
     * Default constructor.
     */
    public Conv2d() {
        this(new Conv2dConf());
    }

    /**
     * Constructor.
     */
    public Conv2d(int[] filters, int[] strides, int k) {
        this(new Conv2dConf().setFilters(filters).setStrides(strides).setK(k));
    }

    /**
     * Constructor.
     */
    public Conv2d(int[] filters, int[] strides) {
        this(new Conv2dConf().setFilters(filters).setStrides(strides));
    }

    /**
     * Constructor.
     * @param conf the layer's configuration.
     */
    public Conv2d(Conv2dConf conf) {
        super(GPUNode.NODES_PATH, "conv_2d.cu", new String[]{"activation", "inputs_gradients", "weights_gradients"});
        if (conf.filters().length != 3)
            throw new RuntimeException("Invalid filters's size: expected 3 got " + conf.filters().length);
        if (conf.strides().length != 2)
            throw new RuntimeException("Invalid strides's size: expected 2 got " + conf.strides().length);
        if (conf.useKWTA() && (conf.k() <= 0 || conf.k() > conf.filters()[0]))
            throw new RuntimeException("Invalid k: expected value between one and the number of filters got " + conf.k());
        if (conf.useCPCA() && (conf.ratio() < 0 || conf.ratio() > 1))
            throw new RuntimeException("Invalid ratio: expected value between zero and one got " + conf.ratio());
        this.conf = conf;
        this.activation = NodesFactory.create("Activation", "gpu", conf.getAf());
        this.kwta = NodesFactory.create("KWTA2d", "gpu", conf.k());
    }

    /**
     * Constructor.
     * @param conf the layer's configuration.
     */
    public Conv2d(Object conf) {
        this((Conv2dConf) conf);
    }

    /**
     * Create randomly initialized weights.
     * @param xShape the input's shape.
     */
    private void createWeights(long[] xShape) {
        long[] shape = new long[]{conf.filters()[0], xShape[1], conf.filters()[1], conf.filters()[2]};
        wgpu.copy(Nd4j.rand(shape).mul(2).sub(1));
        shape = new long[]{conf.filters()[0]};
        bwgpu.copy(Nd4j.rand(shape).mul(2).sub(1));
    }

    /**
     * Compute the output shape.
     * @param conf the layer configuration.
     * @param xShape the input' shape.
     * @return the output shape.
     */
    private long[] computeOutputShape(Conv2dConf conf, long[] xShape) {
        // Compute the number of vertical and horizontal position.
        long nr = xShape[2] - conf.filters()[1] + 1;
        nr = (long) Math.ceil(((double)nr) / ((double)conf.strides()[0]));
        long nc = xShape[3] - conf.filters()[2] + 1;
        nc = (long) Math.ceil(((double)nc) / ((double)conf.strides()[1]));
        // Format the output.
        return new long[]{xShape[0], conf.filters()[0], nr, nc};
    }

    /**
     * Compute the output size of the layer.
     * @param conf the layer configuration.
     * @param xShape the input' shape.
     * @return the output size.
     */
    private int computeOutputSize(Conv2dConf conf, long[] xShape) {
        long[] shape = computeOutputShape(conf, xShape);
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
    private int[] createConf(Conv2dConf conf, long[] wshape, long[] xshape, int nbElements, long[] yshape) {
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
     */
    private void activationKernel(Conv2dConf conf, ArrayPtr x, ArrayPtr w, ArrayPtr bw) {
        // Allocate the output and configuration on device memory.
        long[] shape = computeOutputShape(conf, x.getShape());
        int size = computeOutputSize(conf, x.getShape());
        if (yagpu.isNull())
            yagpu = ArrayPtrFactory.empty(shape, Sizeof.FLOAT);
        yagpu.setShape(shape);
        cgpu.copy(createConf(conf, w.getShape(), x.getShape(), size, shape));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), w.toPTR(), bw.toPTR(), yagpu.toPTR());
        execute(
                "activation", parameters,
                (int)shape[1], (int)shape[2], (int)shape[3],
                (int)shape[0], 1, 1,
                (int) shape[0] * Sizeof.FLOAT
        );
    }

    /**
     * Compute the gradients with respect to the inputs.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     */
    private void inputsGradients(Conv2dConf conf, long[] yShape, ArrayPtr w, ArrayPtr g) {
        // Allocate the output and configuration on device memory.
        int size = computeOutputSize(yShape);
        if (yigpu.isNull())
            yigpu = ArrayPtrFactory.empty(yShape, Sizeof.FLOAT);
        yigpu.setShape(yShape);
        cgpu.copy(createConf(conf, w.getShape(), yShape, size, g.getShape()));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), w.toPTR(), g.toPTR(), yigpu.toPTR());
        execute("inputs_gradients", parameters, size, 256, 256 * Sizeof.FLOAT);
    }

    /**
     * Compute the gradient with respect to the weights.
     * @param conf the layer configuration.
     * @param dwShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     */
    private void weightsGradients(Conv2dConf conf, long[] dwShape, ArrayPtr x, ArrayPtr g) {
        // Allocate the output and configuration on device memory.
        int size = computeOutputSize(dwShape);
        int[] config = createConf(conf, dwShape, x.getShape(), size, g.getShape());
        if (ywgpu.isNull())
            ywgpu = ArrayPtrFactory.empty(dwShape, Sizeof.FLOAT);
        ywgpu.setShape(dwShape);
        cgpu.copy(config);
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), g.toPTR(), ywgpu.toPTR());
        execute(
                "weights_gradients", parameters,
                config[13] * config[14], config[15], config[16],
                5, 10, 10,
                500 * Sizeof.FLOAT
        );
    }

    /**
     * Getter.
     * @return the bias weights.
     */
    public INDArray getBw() {
        return bwgpu.toCPU();
    }

    /**
     * Setter.
     * @param bw the bias weights.
     */
    public void setBw(ArrayPtr bw) {
        bwgpu.copy(bw);
    }

    /**
     * Getter.
     * @return the weights.
     */
    public INDArray getW() {
        return wgpu.toCPU();
    }

    /**
     * Setter.
     * @param w the weights.
     */
    public void setW(ArrayPtr w) {
        wgpu.copy(w);
    }

    /**
     * Apply the required padding.
     * @param x the input.
     * @return the padded input.
     */
    private ArrayPtr applyPadding(ArrayPtr x) {
        int expectedSizeY = (int) (conf.filters()[1] + conf.strides()[0] * (x.getShape()[2] - 1));
        Node paddingY = NodesFactory.create("Pad2d", "gpu", new PadConf(2, expectedSizeY, 0f));
        int expectedSizeX = (int) (conf.filters()[2] + conf.strides()[1] * (x.getShape()[3] - 1));
        Node paddingX = NodesFactory.create("Pad2d", "gpu", new PadConf(3, expectedSizeX, 0f));
        return paddingX.activation(false, paddingY.activation(false, x));
    }

    /**
     * Compute the activation of the layer.
     * @param x the input.
     * @param training true if training mode and false otherwise.
     * @return the activation.
     */
    private ArrayPtr activation(ArrayPtr x, boolean training) {
        if (wgpu.isNull() || bwgpu.isNull())
            createWeights(x.getShape());
        if (training)
            inputShape = x.getShape().clone();
        if (conf.padding() == SAME)
            x = applyPadding(x);
        activationKernel(conf, x, wgpu, bwgpu);
        if (conf.useKWTA())
            yagpu = kwta.activation(training, yagpu);
        if (training)
            input = x;
        yagpu = activation.activation(training, yagpu);
        return yagpu;
    }

    /**
     * Compute the activation of the layer.
     * @param training the mode (training vs testing).
     * @param x is the input.
     * @return the activation.
     */
    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        return activation(x[0], training);
    }

    /**
     * Update the layer's weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the gradient with respect to the input.
     */
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi).
        gradient = activation.update(lr, gradient)[0];
        // Compute the gradient, i.e. inputs, weights, bias weights and cpca.
        inputsGradients(conf, inputShape, wgpu, gradient);
        weightsGradients(conf, wgpu.getShape(), input, gradient);
        INDArray dbw = gradient.toCPU().sum(0, 2, 3);
        INDArray dwcpca;
        if (conf.useCPCA()) {
            dwcpca = cpca.weightsGradients(conf, input, wgpu, yagpu);
            op.mul(ywgpu, (float)(1 - conf.ratio()));
            op.add(ywgpu, ArrayPtrFactory.fromData(dwcpca.mul(conf.ratio())));
            dbw = dbw.mul(1 - conf.ratio());
        }
        // Update the weights.
        op.mul(ywgpu, (float)(-1 * lr));
        op.add(wgpu, ywgpu);
        op.add(bwgpu, ArrayPtrFactory.fromData(dbw.mul(-1 * lr), true));
        return yigpu;
    }

    /**
     * Update the layer's weights.
     * @param lr the learning rate.
     * @param gradient the back propagation gradient from the upper layer.
     * @return the gradient with respect to the input.
     */
    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        if (gradient.length != 1)
            throw new InvalidParameterException();
        return new ArrayPtr[]{update(gradient[0], lr)};
    }

    /**
     * Save the layer.
     * @param kryo the kryo object.
     * @param output the kryo output.
     */
    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Conv2d");
        conf.save(kryo, output);
        kryo.writeObject(output, wgpu.toCPU().data().asFloat());
        kryo.writeObject(output, wgpu.getShape());
        kryo.writeObject(output, bwgpu.toCPU().data().asFloat());
        kryo.writeObject(output, bwgpu.getShape());
        activation.save(kryo, output);
    }

    /**
     * Load the layer's weights.
     * @param kryo the kryo object.
     * @param input the kryo input.
     * @return this.
     */
    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        conf.loadWeights(kryo, input);
        wgpu = ArrayPtrFactory.fromData(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        bwgpu = ArrayPtrFactory.fromData(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        activation.loadWeights(kryo, input);
        return this;
    }

    /**
     * Load the layer.
     * @param kryo the kryo object.
     * @param input the kryo input.
     * @return this.
     */
    @Override
    public Node load(Kryo kryo, Input input) {
        conf = new Conv2dConf();
        conf.load(kryo, input);
        wgpu = ArrayPtrFactory.fromData(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        bwgpu = ArrayPtrFactory.fromData(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        activation.load(kryo, input);
        return this;
    }

    /**
     * Print the layer.
     */
    @Override
    public void print() {
        System.out.println("Type: Conv2d(gpu)");
        System.out.println("Filters: " + Arrays.toString(wgpu.getShape()));
        System.out.println("Strides: " + Arrays.toString(conf.strides()));
        System.out.println("Number of winners in kWTA: " + conf.k());
        System.out.println("Padding: " + conf.padding());
        System.out.println("Ratio (Back-propagation vs Hebbian): " + conf.ratio());
    }
}
