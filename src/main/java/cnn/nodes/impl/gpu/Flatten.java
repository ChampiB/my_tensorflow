package cnn.nodes.impl.gpu;

import cnn.nodes.Node;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.security.InvalidParameterException;

/**
 * Flatten layer.
 */
public class Flatten extends Node {

    private long[] shape = null;

    public ArrayPtr activation(ArrayPtr x) {
        if (shape == null)
            shape = x.getShape();
        long bs = shape[0];
        long res = 1;
        for (int i = 1; i < shape.length; i++)
            res *= shape[i];
        x.setShape(new long[]{bs, res});
        return x;
    }

    /**
     * Compute the layer activation.
     *
     * @param x        is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        return activation(x[0]);
    }

    public ArrayPtr update(ArrayPtr gradient, double lr) {
        if (shape != null)
            gradient.setShape(shape);
        return gradient;
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        if (gradient.length != 1)
            throw new InvalidParameterException();
        return new ArrayPtr[]{update(gradient[0], lr)};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Flatten");
        kryo.writeObject(output, shape);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, long[].class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        shape = kryo.readObject(input, long[].class);
        return this;
    }

    /**
     * Display the layer on the standard output.
     */
    public void print() {
        System.out.println("Type: Flatten (gpu)");
    }
}
