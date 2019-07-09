package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.conf.PadConf;
import cnn.data.ArrayPtr;
import cnn.useful.gpu.GPUNode;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.security.InvalidParameterException;

public class Pad2d extends GPUNode {

    private PadConf conf;
    private long[] xcShape;
    private ArrayPtr xShape = ArrayPtrFactory.empty();
    private ArrayPtr yShape = ArrayPtrFactory.empty();
    private ArrayPtr ya = ArrayPtrFactory.empty();
    private ArrayPtr yi = ArrayPtrFactory.empty();

    public Pad2d() {
        this(new PadConf());
    }

    public Pad2d(PadConf conf) {
        super(GPUNode.NODES_PATH, "pad_2d.cu", new String[]{"activation", "inputs_gradients"});
        this.conf = conf;
    }

    public Pad2d(Object conf) {
        this((PadConf) conf);
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        xcShape = x[0].getShape();
        if (x.length != 1 || conf.getExpectedSize() < xcShape[conf.getDim()])
            throw new InvalidParameterException();
        if (xShape.isNull())
            xShape = ArrayPtrFactory.fromData(xcShape, true);
        long[] ycShape = xcShape.clone();
        ycShape[conf.getDim()] = conf.getExpectedSize();
        if (yShape.isNull())
            yShape = ArrayPtrFactory.fromData(ycShape, true);
        if (ya.isNull())
            ya = ArrayPtrFactory.empty(ycShape, Sizeof.FLOAT);
        Pointer parameters = Pointer.to(xShape.toPTR(), x[0].toPTR(), yShape.toPTR(), ya.toPTR(), Pointer.to(new float[]{conf.getPadValue()}));
        execute(
                "activation", parameters,
                (int)ycShape[1], (int)ycShape[2], (int)ycShape[3],
                (int)ycShape[0], 1, 1,
                0
        );
        return ya;
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        if (gradient.length != 1)
            throw new InvalidParameterException();
        if (xShape.isNull())
            xShape = ArrayPtrFactory.fromData(gradient[0].getShape(), true);
        if (yShape.isNull())
            yShape = ArrayPtrFactory.fromData(gradient[0].getShape(), true);
        if (yi.isNull())
            yi = ArrayPtrFactory.empty(xcShape, Sizeof.FLOAT);
        Pointer parameters = Pointer.to(yShape.toPTR(), gradient[0].toPTR(), xShape.toPTR(), yi.toPTR());
        execute(
                "inputs_gradients", parameters,
                (int) xcShape[1], (int) xcShape[2], (int) xcShape[3],
                (int) xcShape[0], 1, 1,
                0
        );
        return new ArrayPtr[]{yi};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Pad2d");
        conf.save(kryo, output);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        conf.loadWeights(kryo, input);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        conf.load(kryo, input);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Pad2d(gpu)");
        System.out.println("Dimension: " + conf.getDim());
        System.out.println("Expected output size: " + conf.getExpectedSize());
        System.out.println("Padding value: " + conf.getPadValue());
    }
}
