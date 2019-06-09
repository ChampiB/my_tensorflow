package cnn.layers;

import cnn.layers.activation.Activation;
import cnn.layers.perf.DenseInterface;
import cnn.layers.perf.TasksFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Fully connected layer.
 */
public class Dense extends Layer {

    private int outputSize;
    private INDArray w = null;
    private INDArray input = null;
    private INDArray z = null;
    private Activation.Type af;
    private DenseInterface dense;

    /**
     * Create a fully connected layer.
     * @param outputSize the number of neurons in the layer.
     * @param af the activation function of the layer.
     */
    public Dense(int outputSize, Activation.Type af) {
        this.af = af;
        this.outputSize = outputSize;
        this.dense = TasksFactory.create("Dense");
    }

    /**
     * Create a fully connected layer.
     * @param outputSize the number of neurons in the layer.
     */
    public Dense(int outputSize) {
        this(outputSize, Activation.Type.SIGMOID);
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
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    public INDArray activation(INDArray x, boolean training) {
        if (w == null)
            w = createWeights(x.shape());
        x = Nd4j.concat(1, Nd4j.ones(new long[]{(int)x.shape()[0], 1}), x);
        if (training)
            input = x;
        z = dense.activation(x, w);
        return Activation.get(af).apply(z);
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    public INDArray update(INDArray gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi)
        gradient = gradient.mul(Activation.derivative(af).apply(z));
        // Compute the derivative of cost function with respect to the inputs
        INDArray di = dense.inputsGradients(input.shape(), w, gradient);
        // Compute the derivative of cost function with respect to the weights and update the weights
        w = w.add(dense.weightsGradients(w.shape(), input, gradient).mul(-1 * lr));
        return di;
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
    public void setW(INDArray w) {
        this.w = w;
    }

    /**
     * Display the layer on the standard output.
     */
    public void print() {
        System.out.println("Type: Dense");
        System.out.println("Size: " + outputSize);
        System.out.println();
    }
}
