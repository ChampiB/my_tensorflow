package cnn.layers.impl.cpu;

import cnn.layers.Layer;
import cnn.ops.ActivationInterface;
import cnn.ops.OpsFactory;
import cnn.ops.cpu.Activation;
import cnn.layers.conf.DenseConf;
import cnn.useful.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dense implements Layer {

    private int outputSize;
    private INDArray w = Nd4j.empty();
    private INDArray input = Nd4j.empty();
    private INDArray z = Nd4j.empty();
    private Activation.Type af;

    private ActivationInterface layerActivation = OpsFactory.create("Activation", "cpu");

    /**
     * Default constructor.
     */
    public Dense() {}

    /**
     * Create a fully connected layer.
     * @param nbOutputs the number of outputs of the layer.
     */
    public Dense(int nbOutputs) {
        this(new DenseConf(nbOutputs));
    }

    /**
     * Create a fully connected layer.
     * @param nbOutputs the number of outputs of the layer.
     * @param af the activation of the layer.
     */
    public Dense(int nbOutputs, Activation.Type af) {
        this(new DenseConf(nbOutputs, af));
    }

    /**
     * Create a fully connected layer.
     * @param conf the layer's configuration.
     */
    public Dense(DenseConf conf) {
        this.outputSize = conf.getOutputSize();
        this.af = conf.getAf();
    }

    /**
     * Constructor.
     * @param conf the layer's configuration.
     */
    public Dense(Object conf) {
        this((DenseConf) conf);
    }

    /**
     * Remove the first column corresponding to the bais input.
     * @param a the array.
     * @param ishape the input shape.
     * @return the new array without the bias.
     */
    private INDArray removeBiasColumn(INDArray a, long[] ishape) {
        long[] offset = new long[]{(int) ishape[0], 0};
        int[] shape = new int[]{(int) ishape[0], (int) ishape[1] - 1};
        int[] stride = new int[]{1, (int) ishape[0]};
        return a.subArray(offset, shape, stride);
    }

    /**
     * Create randomly initialized weights.
     * @param xShape the input's shape.
     * @return the array of weights.
     */
    private INDArray createWeights(long[] xShape) {
        long[] shape = new long[]{xShape[1] + 1, outputSize};
        return Nd4j.rand(shape).mul(2).sub(1);
    }

    /**
     * Getter.
     * @return the layer's weights.
     */
    public INDArray getW() {
        return w;
    }

    /**
     * Setter.
     * @param w the layer's weights.
     */
    public void setW(ArrayPtr w) {
        this.w = w.toCPU().dup();
    }

    @Override
    public ArrayPtr activation(ArrayPtr x, boolean training) {
        if (w.isEmpty())
            w = createWeights(x.toCPU().shape());
        INDArray input = Nd4j.concat(1, Nd4j.ones(new long[]{(int)x.toCPU().shape()[0], 1}), x.toCPU());
        if (training)
            this.input = input;
        z = input.mmul(w);
        return layerActivation.apply(af, new ArrayPtr(z));
    }

    @Override
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi)
        INDArray grad = gradient.toCPU().mul(layerActivation.derivative(af, new ArrayPtr(z)).toCPU());
        // Compute the derivative of cost function with respect to the inputs
        INDArray di = removeBiasColumn(grad.mmul(w.transpose()), input.shape());
        // Compute the derivative of cost function with respect to the weights and update the weights
        w = w.add(input.transpose().mmul(grad).mul(-1 * lr));
        return new ArrayPtr(di);
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Dense");
        kryo.writeObject(output, outputSize);
        kryo.writeObject(output, af);
        kryo.writeObject(output, w.data().asFloat());
        kryo.writeObject(output, w.shape());
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        outputSize = kryo.readObject(input, int.class);
        kryo.readObject(input, ActivationInterface.Type.class);
        w = Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class));
        return this;
    }

    @Override
    public Layer load(Kryo kryo, Input input) {
        outputSize = kryo.readObject(input, int.class);
        af = kryo.readObject(input, ActivationInterface.Type.class);
        w = Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class));
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Dense(cpu)");
        System.out.println("Size: " + outputSize);
        System.out.println();
    }
}
