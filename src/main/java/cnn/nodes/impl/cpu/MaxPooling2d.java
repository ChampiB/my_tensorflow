package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.conf.Pooling2dConf;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.security.InvalidParameterException;

public class MaxPooling2d extends Node {

    private int[] kernel;
    private INDArray mask;

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        this(new Pooling2dConf());
    }

    /**
     * Default constructor.
     * @param kernel the size of the pooling kernel.
     */
    public MaxPooling2d(int[] kernel) {
        this(new Pooling2dConf(kernel));
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public MaxPooling2d(Pooling2dConf conf) {
        this.kernel = conf.getKernel();
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public MaxPooling2d(Object conf) {
        this((Pooling2dConf) conf);
    }

    private ArrayPtr activation(ArrayPtr x, boolean training) {
        long[] shape = x.getShape();
        INDArray y = Nd4j.zeros(shape[0], shape[1], shape[2] / kernel[0], shape[3] / kernel[1]);
        mask = (training) ? Nd4j.zeros(shape) : Nd4j.zeros();
        for (int ii = 0; ii < y.shape()[0]; ii++) {
            for (int fi = 0; fi < y.shape()[1]; fi++) {
                for (int vi = 0; vi < y.shape()[2]; vi++) {
                    for (int hi = 0; hi < y.shape()[3]; hi++) {
                        INDArray xKernel = x.toCPU().get(
                                NDArrayIndex.point(ii), NDArrayIndex.point(fi),
                                NDArrayIndex.interval(vi * kernel[0], (vi + 1) * kernel[0]),
                                NDArrayIndex.interval(hi * kernel[1], (hi + 1) * kernel[1])
                        );
                        double maxValue = xKernel.maxNumber().doubleValue();
                        y.putScalar(ii, fi, vi, hi, maxValue);
                        if (training) {
                            int idx = Nd4j.getExecutioner()
                                    .execAndReturn(new IAMax(xKernel)).getFinalResult().intValue();
                            idx = (int) (ii * shape[1] * shape[2] * shape[3] +
                                    fi * shape[2] * shape[3] +
                                    (vi * kernel[0] + idx / kernel[1]) * shape[3] +
                                    hi * kernel[1] +
                                    idx % kernel[1]);
                            mask.putScalar(idx, 1);
                        }
                    }
                }
            }
        }
        return ArrayPtrFactory.fromData(y);
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        if (x.length != 1)
            throw new InvalidParameterException();
        return activation(x[0], training);
    }

    public ArrayPtr update(ArrayPtr gradient) {
        for (int ii = 0; ii < gradient.getShape()[0]; ii++) {
            for (int fi = 0; fi < gradient.getShape()[1]; fi++) {
                for (int vi = 0; vi < gradient.getShape()[2]; vi++) {
                    for (int hi = 0; hi < gradient.getShape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[]{
                                NDArrayIndex.point(ii), NDArrayIndex.point(fi),
                                NDArrayIndex.interval(vi * kernel[0], (vi + 1) * kernel[0]),
                                NDArrayIndex.interval(hi * kernel[1], (hi + 1) * kernel[1])
                        };
                        INDArray xKernel = mask.get(indexes).mul(gradient.toCPU().getNumber(ii, fi, vi, hi));
                        mask.put(indexes, xKernel);
                    }
                }
            }
        }
        return ArrayPtrFactory.fromData(mask);
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        if (gradient.length != 1)
            throw new InvalidParameterException();
        return new ArrayPtr[]{update(gradient[0])};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "MaxPooling2d");
        kryo.writeObject(output, kernel);
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: MaxPooling2d(cpu)");
    }
}
