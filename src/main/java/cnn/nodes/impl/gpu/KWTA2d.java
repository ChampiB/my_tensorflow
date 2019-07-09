package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.ops.OperationInterface;
import cnn.ops.OpsFactory;
import cnn.data.ArrayPtr;
import cnn.useful.gpu.GPUNode;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.security.InvalidParameterException;

public class KWTA2d extends GPUNode {

    private ArrayPtr y = ArrayPtrFactory.empty();
    private ArrayPtr c = ArrayPtrFactory.empty();
    private ArrayPtr m = ArrayPtrFactory.empty();

    private OperationInterface op;
    private int k;

    /**
     * Default constructor.
     */
    public KWTA2d() {
        this(1);
    }

    public KWTA2d(Integer k) {
        super(GPUNode.NODES_PATH, "kwta_2d.cu", new String[]{"activation"});
        this.op = OpsFactory.create("Operation", "gpu");
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
     * Create the configuration.
     * @param shape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(long[] shape, int nbElements) {
        return new int[]{
                (int)(shape[1] * shape[2] * shape[3]), (int)(shape[2] * shape[3]), (int)shape[3], nbElements
        };
    }

    /**
     * Compute the k-winners-take-all activation.
     * @param x the input.
     * @return the output.
     */
    public ArrayPtr activation(ArrayPtr x) {
        // Allocate the output and configuration on device memory.
        if (y.isNull())
            y = ArrayPtrFactory.empty(x.getShape(), Sizeof.FLOAT);
        if (m.isNull())
            m = ArrayPtrFactory.empty(x.getShape(), Sizeof.FLOAT);
        y.copy(x);
        c.copy(createConf(x.getShape(), x.getSize()));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(c.toPTR(), Pointer.to(new int[]{k}), y.toPTR(), x.toPTR(), m.toPTR());
        execute(
                "activation", parameters,
                (int)x.getShape()[2], (int)x.getShape()[3], 1,
                (int)x.getShape()[0], 1, 1,
                0
        );
        return x;
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        return activation(x[0]);
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        op.mul(m, gradient[0]);
        return new ArrayPtr[]{m.dup()};
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
        System.out.println("Type: KWTA2d(gpu)");
        System.out.println("K: " + k);
    }
}
