package cnn.nodes.conf;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

public class PadConf {

    private int dim;
    private int expectedSize;
    private float padValue;

    public PadConf() {
        this(0, 0, 0);
    }

    public PadConf(int dim, int expectedSize, float padValue) {
        this.dim = dim;
        this.expectedSize = expectedSize;
        this.padValue = padValue;
    }

    public float getPadValue() {
        return padValue;
    }

    public int getDim() {
        return dim;
    }

    public int getExpectedSize() {
        return expectedSize;
    }

    public PadConf setDim(int dim) {
        this.dim = dim;
        return this;
    }

    public PadConf setExpectedSize(int expectedSize) {
        this.expectedSize = expectedSize;
        return this;
    }

    public PadConf setPadValue(float padValue) {
        this.padValue = padValue;
        return this;
    }

    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, dim);
        kryo.writeObject(output, expectedSize);
        kryo.writeObject(output, padValue);
    }

    public PadConf loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, Integer.class);
        kryo.readObject(input, Integer.class);
        kryo.readObject(input, Float.class);
        return this;
    }

    public PadConf load(Kryo kryo, Input input) {
        dim = kryo.readObject(input, Integer.class);
        expectedSize = kryo.readObject(input, Integer.class);
        padValue = kryo.readObject(input, Float.class);
        return this;
    }
}
