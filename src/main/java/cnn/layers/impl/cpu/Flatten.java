package cnn.layers.impl.cpu;

import cnn.layers.Layer;
import cnn.useful.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

/**
 * Flatten layer.
 */
public class Flatten implements Layer {

    private long[] shape = null;

    /**
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    public ArrayPtr activation(ArrayPtr x, boolean training) {
        if (shape == null)
            shape = x.getShape();
        long bs = shape[0];
        long res = 1;
        for (int i = 1; i < shape.length; i++)
            res *= shape[i];
        return new ArrayPtr(x.toCPU().reshape(bs, res));
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        return new ArrayPtr((shape == null) ? gradient.toCPU().reshape(): gradient.toCPU().reshape(shape));
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Flatten");
        kryo.writeObject(output, shape);
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, long[].class);
        return this;
    }

    @Override
    public Layer load(Kryo kryo, Input input) {
        shape = kryo.readObject(input, long[].class);
        return this;
    }

    /**
     * Display the layer on the standard output.
     */
    public void print() {
        System.out.println("Type: Flatten");
        System.out.println();
    }
}
