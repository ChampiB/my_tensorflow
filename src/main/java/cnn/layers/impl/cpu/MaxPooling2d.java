package cnn.layers.impl.cpu;

import cnn.layers.Layer;
import cnn.layers.conf.MaxPooling2dConf;
import cnn.useful.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MaxPooling2d implements Layer {

    private int[] kernel;
    private INDArray mask;

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        this(new MaxPooling2dConf());
    }

    /**
     * Default constructor.
     * @param kernel the size of the pooling kernel.
     */
    public MaxPooling2d(int[] kernel) {
        this(new MaxPooling2dConf(kernel));
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public MaxPooling2d(MaxPooling2dConf conf) {
        this.kernel = conf.getKernel();
    }

    /**
     * Default constructor.
     * @param conf the layer's configuration.
     */
    public MaxPooling2d(Object conf) {
        this((MaxPooling2dConf) conf);
    }

    @Override
    public ArrayPtr activation(ArrayPtr x, boolean training) {
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
        return new ArrayPtr(y);
    }

    @Override
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        for (int ii = 0; ii < gradient.toCPU().shape()[0]; ii++) {
            for (int fi = 0; fi < gradient.toCPU().shape()[1]; fi++) {
                for (int vi = 0; vi < gradient.toCPU().shape()[2]; vi++) {
                    for (int hi = 0; hi < gradient.toCPU().shape()[3]; hi++) {
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
        return new ArrayPtr(mask);
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "MaxPooling2d");
        kryo.writeObject(output, kernel);
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public Layer load(Kryo kryo, Input input) {
        kernel = kryo.readObject(input, int[].class);
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: MaxPooling2d(cpu)");
        System.out.println();
    }
}
