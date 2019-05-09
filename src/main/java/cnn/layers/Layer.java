package cnn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Generic layer abstraction.
 */
public abstract class Layer {
    /**
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    public abstract INDArray activation(INDArray x, boolean training);

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    public abstract INDArray update(INDArray gradient, double lr);

    /**
     * Display the layer on the standard output.
     */
    public abstract void print();
}
