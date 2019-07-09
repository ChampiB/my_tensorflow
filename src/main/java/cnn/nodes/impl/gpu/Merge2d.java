package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Sizeof;

public class Merge2d extends Node {

    private long[][] xShapes = null;
    private long[] yShape = null;
    private ArrayPtr y = ArrayPtrFactory.empty();

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... xs) {
        if (xShapes ==  null) {
            xShapes = new long[xs.length][];
            yShape = xs[0].getShape().clone();
            yShape[1] = 0;
            for (int i = 0; i < xs.length; i++) {
                xShapes[i] = xs[i].getShape().clone();
                yShape[1] += xs[i].getShape()[1];
            }
        }
        if (y.isNull())
            y = ArrayPtrFactory.empty(yShape, Sizeof.FLOAT);
        int offset = 0;
        for (ArrayPtr x : xs) {
            y.copyDToD(x, offset);
            offset += x.getSize();
        }
        return y;
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        ArrayPtr[] result = new ArrayPtr[xShapes.length];
        int sOffset = 0;
        for (int i = 0; i < xShapes.length; i++) {
            result[i] = ArrayPtrFactory.empty(xShapes[i], Sizeof.FLOAT);
            int dOffset = 0;
            for (int j = 0; j < xShapes[0][0]; j++) {
                long size = xShapes[i][1] * xShapes[i][2] * xShapes[i][3];
                result[i].copyDToD(gradient[0], dOffset, sOffset, size);
                dOffset += size;
                sOffset += size;
            }
        }
        return result;
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Merge2d");
        kryo.writeObject(output, xShapes);
        kryo.writeObject(output, yShape);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, long[][].class);
        kryo.readObject(input, long[].class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        xShapes = kryo.readObject(input, long[][].class);
        yShape = kryo.readObject(input, long[].class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Merge2d (gpu)");
    }
}
