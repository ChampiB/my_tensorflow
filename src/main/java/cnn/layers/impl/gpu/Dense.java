package cnn.layers.impl.gpu;

import cnn.layers.Layer;
import cnn.ops.OperationInterface;
import cnn.ops.cpu.Activation;
import cnn.layers.conf.DenseConf;
import cnn.ops.ActivationInterface;
import cnn.ops.OpsFactory;
import cnn.useful.ArrayPtr;
import cnn.useful.gpu.GPUTask;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dense extends GPUTask implements Layer {

    private ArrayPtr c = new ArrayPtr();
    private ArrayPtr x = new ArrayPtr();
    private ArrayPtr w = new ArrayPtr();
    private ArrayPtr z = new ArrayPtr();
    private ArrayPtr ya = new ArrayPtr();
    private ArrayPtr yw = new ArrayPtr();
    private ArrayPtr yi = new ArrayPtr();

    private int outputSize = 10;
    private Activation.Type af = ActivationInterface.Type.SIGMOID;

    private ActivationInterface layerActivation = OpsFactory.create("Activation", "gpu");
    private OperationInterface op = OpsFactory.create("Operation", "gpu");

    /**
     * Create a fully connected layer.
     * @param nbOutputs the number of outputs of the layer.
     * @param af the activation function of the layer.
     */
    public Dense(int nbOutputs, Activation.Type af) {
        this(new DenseConf(nbOutputs, af));
    }

    /**
     * Default constructor.
     */
    public Dense() {
        super(GPUTask.LAYERS_PATH, "dense.cu", new String[]{"activation", "inputs_gradients", "weights_gradients"});
    }

    /**
     * Create a fully connected layer.
     * @param conf the layer's configuration.
     */
    public Dense(DenseConf conf) {
        this();
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
     * Crete the configuration.
     * @param nImage the number of images.
     * @param nInput the number of input neurons.
     * @param nOutput the number of output neurons.
     * @return the configuration.
     */
    private int[] createConf(long nImage, long nInput, long nOutput) {
        return new int[]{(int)nImage, (int)nInput, (int)nOutput};
    }

    /**
     * Create randomly initialized weights.
     * @param xShape the input's shape.
     * @return the array of weights.
     */
    private INDArray createWeights(long[] xShape) {
        return Nd4j.rand(new long[]{xShape[1] + 1, outputSize}).mul(2).sub(1);
    }

    /**
     * Getter.
     * @return the layer's weights.
     */
    public INDArray getW() {
        return w.toCPU();
    }

    /**
     * Setter.
     * @param w the layer's weights.
     */
    public void setW(ArrayPtr w) {
        this.w.copy(w);
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param x the input.
     * @param w the weights.
     */
    private ArrayPtr activationKernel(ArrayPtr x, ArrayPtr w) {
        // Allocate the output and configuration on device memory.
        if (ya.isNull())
            ya = new ArrayPtr(new long[]{x.getShape()[0], w.getShape()[1]}, Sizeof.FLOAT);
        c.copy(createConf(x.getShape()[0], w.getShape()[0], w.getShape()[1]));
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(c.toPTR(), x.toPTR(), w.toPTR(), ya.toPTR());
        execute(
                "activation", kernelParameters,
                (int)x.getShape()[0], (int)w.getShape()[1], 1,
                512, 1, 1,
                512 * Sizeof.FLOAT
        );
        if (z.isNull())
            z = new ArrayPtr(ya.getShape(), Sizeof.FLOAT);
        z.copy(ya);
        return z;
    }

    /**
     * Compute the gradients with respect to the inputs.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     */
    private void inputsGradientsKernel(long[] yShape, ArrayPtr w, ArrayPtr g) {
        // Allocate the output and configuration on device memory.
        if (yi.isNull())
            yi = new ArrayPtr(new long[]{yShape[0], yShape[1]}, Sizeof.FLOAT);
        c.copy(createConf(yShape[0], yShape[1] + 1, g.getShape()[1]));
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(c.toPTR(), w.toPTR(), g.toPTR(), yi.toPTR());
        execute(
                "inputs_gradients", kernelParameters,
                (int)yShape[0], (int)yShape[1], 1,
                32, 1, 1,
                32 * Sizeof.FLOAT
        );
    }

    /**
     * Compute the gradient with respect to the weights.
     * @param dwShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     */
    private void weightsGradientsKernel(long[] dwShape, ArrayPtr x, ArrayPtr g) {
        // Allocate the output and configuration on device memory.
        if (yw.isNull())
            yw = new ArrayPtr(dwShape, Sizeof.FLOAT);
        c.copy(createConf(x.getShape()[0], dwShape[0], g.getShape()[1]));
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(c.toPTR(), x.toPTR(), g.toPTR(), yw.toPTR());
        execute(
                "weights_gradients", kernelParameters,
                (int)g.getShape()[1], (int)dwShape[0], 1,
                32, 1, 1,
                32 * Sizeof.FLOAT
        );
    }

    @Override
    public ArrayPtr activation(ArrayPtr x, boolean training) {
        if (w.isNull())
            w.copy(createWeights(x.getShape()));
        if (training)
            this.x = x;
        z = activationKernel(x, w);
        return layerActivation.apply(af, ya);
    }

    @Override
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi)
        op.mul(gradient, layerActivation.derivative(af, z));
        // Compute the derivative of cost function with respect to the inputs and the weights.
        inputsGradientsKernel(x.getShape(), w, gradient);
        weightsGradientsKernel(w.getShape(), x, gradient);
        // Update the weights
        op.mul(yw, (float)(-1 * lr));
        op.add(w, yw);
        return yi;
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Dense");
        kryo.writeObject(output, outputSize);
        kryo.writeObject(output, af);
        kryo.writeObject(output, w.toCPU().data().asFloat());
        kryo.writeObject(output, w.getShape());
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        outputSize = kryo.readObject(input, int.class);
        kryo.readObject(input, ActivationInterface.Type.class);
        w = new ArrayPtr(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        return this;
    }

    @Override
    public Layer load(Kryo kryo, Input input) {
        outputSize = kryo.readObject(input, int.class);
        af = kryo.readObject(input, ActivationInterface.Type.class);
        w = new ArrayPtr(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Dense(gpu)");
        System.out.println("Size: " + outputSize);
        System.out.println();
    }
}
