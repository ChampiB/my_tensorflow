package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.conf.PadConf;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.security.InvalidParameterException;

public class Pad2d extends Node {

    private PadConf conf;
    private long[] xShape;

    public Pad2d() {
        this(new PadConf());
    }

    public Pad2d(PadConf conf) {
        this.conf = conf;
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        xShape = x[0].getShape().clone();
        if (conf.getExpectedSize() < xShape[conf.getDim()])
            throw new InvalidParameterException();
        long[] shape = xShape;
        shape[conf.getDim()] = conf.getExpectedSize() - xShape[conf.getDim()];
        INDArray pad = Nd4j.ones(shape).mul(conf.getPadValue());
        return ArrayPtrFactory.fromData(Nd4j.concat(conf.getDim(), x[0].toCPU(), pad));
    }

    public Pad2d(Object conf) {
        this((PadConf) conf);
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        INDArrayIndex[] indexes = new INDArrayIndex[xShape.length];
        for (int i = 0; i < xShape.length; i++)
            indexes[i] = NDArrayIndex.interval(0, xShape[i]);
        return new ArrayPtr[]{ArrayPtrFactory.fromData(gradient[0].toCPU().get(indexes))};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Pad2d");
        kryo.writeObject(output, xShape);
        conf.save(kryo, output);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, long[].class);
        conf.loadWeights(kryo, input);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        xShape = kryo.readObject(input, long[].class);
        conf.load(kryo, input);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Pad2d(cpu)");
        System.out.println("Dimension: " + conf.getDim());
        System.out.println("Expected output size: " + conf.getExpectedSize());
        System.out.println("Padding value: " + conf.getPadValue());
    }
}
