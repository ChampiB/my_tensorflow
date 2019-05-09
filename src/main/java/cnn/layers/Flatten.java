package cnn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Flatten layer.
 */
public class Flatten extends Layer {

    private long[] shape = null;

    /**
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    public INDArray activation(INDArray x, boolean training) {
        if (shape == null)
            shape = x.shape();
        long bs = shape[0];
        long res = 1;
        for (int i = 1; i < shape.length; i++)
            res *= shape[i];
        return x.reshape(bs, res);
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    public INDArray update(INDArray gradient, double lr) {
        return (shape == null) ? gradient.reshape(): gradient.reshape(shape);
    }

    /**
     * Display the layer on the standard output.
     */
    public void print() {
        System.out.println("Type: Flatten");
        System.out.println();
    }
}
