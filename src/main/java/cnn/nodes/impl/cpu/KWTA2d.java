package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.impl.cpu.useful.KWTATask;
import cnn.data.ArrayPtr;
import cnn.useful.cpu.ThreadPool;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.security.InvalidParameterException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Future;

public class KWTA2d extends Node {

    private int k;
    private INDArray mask;

    /**
     * Default constructor.
     */
    public KWTA2d() {
        this(1);
    }

    public KWTA2d(Integer k) {
        this.k = k;
    }

    /**
     * Constructor.
     * @param conf the convolution con
     */
    public KWTA2d(Object conf) {
        this((Integer) conf);
    }

    /**
     * Compute the k-winners-take-all activation.
     * @param x the input.
     * @return the output.
     */
    public ArrayPtr activation(ArrayPtr x) {
        // Launch one convolution task for each image.
        INDArray result = Nd4j.zeros(x.getShape());
        List<Future<Boolean>> tasks = new LinkedList<>();
        for (int ii = 0; ii < x.getShape()[0]; ii++) {
            tasks.add(ThreadPool.getInstance().submit(new KWTATask(x.toCPU(), result, ii, k)));
        }
        ThreadPool.waitAll(tasks);
        mask = result.dup();
        BooleanIndexing.replaceWhere(mask, 1, Conditions.notEquals(0));
        return ArrayPtrFactory.fromData(result);
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        return activation(x[0]);
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        return new ArrayPtr[]{ArrayPtrFactory.fromData(gradient[0].toCPU().mul(mask))};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "KWTA2d");
        kryo.writeObject(output, k);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, Integer.class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        k = kryo.readObject(input, Integer.class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: KWTA2d(cpu)");
        System.out.println("K: " + k);
    }
}
