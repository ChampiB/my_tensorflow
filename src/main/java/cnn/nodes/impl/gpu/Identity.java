package cnn.nodes.impl.gpu;

import cnn.nodes.Node;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

public class Identity extends Node {

    public Identity() {}

    public Identity(Object conf) {
        this();
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        return x[0];
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        return gradient;
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Identity");
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Identity(gpu)");
    }
}
