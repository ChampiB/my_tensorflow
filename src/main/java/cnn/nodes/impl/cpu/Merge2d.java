package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Merge2d extends Node {

    private long[][] xShapes = null;
    private long[] yShape = null;

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (xShapes == null && training) {
            xShapes = new long[x.length][];
            yShape = x[0].getShape().clone();
            for (int i = 0; i < x.length; i++) {
                xShapes[i] = x[i].getShape().clone();
                if (i != 0)
                    yShape[1] += x[i].getShape()[1];
            }
        }
        INDArray[] arrays = new INDArray[x.length];
        for (int i = 0; i < x.length; i++) {
            arrays[i] = x[i].toCPU();
        }
        return ArrayPtrFactory.fromData(Nd4j.concat(1, arrays));
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        ArrayPtr[] result = new ArrayPtr[xShapes.length];
        int index = 0;
        for (int i = 0; i < xShapes.length; i++) {
            result[i] = ArrayPtrFactory.fromData(gradient[0].toCPU().get(
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(index, index + xShapes[i][1]),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            ));
            index += xShapes[i][1];
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
        System.out.println("Type: Merge2d (cpu)");
    }
}
