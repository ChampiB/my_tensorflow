package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.ops.OperationInterface;
import cnn.ops.OpsFactory;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.factory.Nd4j;

import java.security.InvalidParameterException;

public class Add extends Node {

    private OperationInterface op;
    private int nbInputs;

    public Add() {
        op = OpsFactory.create("Operation", "cpu");
    }

    public Add(Object conf) {
        this();
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... xs) {
        if (xs.length == 0)
            throw new InvalidParameterException();
        nbInputs = xs.length;
        ArrayPtr result = ArrayPtrFactory.fromData(Nd4j.zeros(xs[0].getShape()));
        for (ArrayPtr x : xs)
            op.add(result, x);
        return result;
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        if (gradient.length != 1)
            throw new InvalidParameterException();
        ArrayPtr[] result = new ArrayPtr[nbInputs];
        for (int i = 0; i < nbInputs; i++)
            result[i] = gradient[0].dup();
        return result;
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Add");
        kryo.writeObject(output, nbInputs);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, Integer.class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        nbInputs = kryo.readObject(input, Integer.class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Add(cpu)");
    }
}
