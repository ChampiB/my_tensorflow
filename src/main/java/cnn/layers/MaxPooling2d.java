package cnn.layers;

import cnn.layers.perf.MaxPooling2dInterface;
import cnn.layers.perf.TasksFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Max pooling layer.
 */
public class MaxPooling2d extends Layer {

    private int[] kernel;
    private INDArray mask;
    private MaxPooling2dInterface maxPooling;

    /**
     * Constructor.
     * @param kernel the kernel size (i.e. pooling size).
     */
    public MaxPooling2d(int[] kernel) {
        this.maxPooling = TasksFactory.create("MaxPooling2d");
        this.kernel = kernel;
    }

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        this(new int[]{2, 2});
    }

    /**
     * Compute the layer activation.
     * @param x is the input [batches, filters, rows, cols].
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    @Override
    public INDArray activation(INDArray x, boolean training) {
        Pair<INDArray, INDArray> result = maxPooling.maxPooling2d(kernel, x, training);
        mask = result.getValue();
        return result.getKey();
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    @Override
    public INDArray update(INDArray gradient, double lr) {
        return maxPooling.inputsGradients(kernel, gradient, mask);
    }

    /**
     * Display the layer on the standard output.
     */
    @Override
    public void print() {
        System.out.println("Type: MaxPooling2d");
        System.out.println();
    }
}
