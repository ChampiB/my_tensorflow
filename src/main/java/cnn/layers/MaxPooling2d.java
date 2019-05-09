package cnn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * Max pooling layer.
 */
public class MaxPooling2d extends Layer {

    private int[] kernel;
    private INDArray mask;

    /**
     * Constructor.
     * @param kernel the kernel size (i.e. pooling size).
     */
    public MaxPooling2d(int[] kernel) {
        this.kernel = kernel;
    }

    /**
     * Default constructor.
     */
    public MaxPooling2d() {
        this(new int[]{2, 2});
    }

    /**
     * Compute the layer activation.
     * @param x is the input [batches, filters, rows, cols].
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    @Override
    public INDArray activation(INDArray x, boolean training) {
        long[] shape = x.shape();
        INDArray y = Nd4j.zeros(shape[0], shape[1], shape[2] / kernel[0], shape[3] / kernel[1]);
        if (training)
            mask = Nd4j.zeros(shape);
        for (int ii = 0; ii < y.shape()[0]; ii++) {
            for (int fi = 0; fi < y.shape()[1]; fi++) {
                for (int vi = 0; vi < y.shape()[2]; vi++) {
                    for (int hi = 0; hi < y.shape()[3]; hi++) {
                        INDArray xKernel = x.get(
                                NDArrayIndex.point(ii), NDArrayIndex.point(fi),
                                NDArrayIndex.interval(vi * kernel[0], (vi + 1) * kernel[0]),
                                NDArrayIndex.interval(hi * kernel[1], (hi + 1) * kernel[1])
                        );
                        double maxValue = xKernel.maxNumber().doubleValue();
                        y.putScalar(ii, fi, vi, hi, maxValue);
                        if (training) {
                            int idx = Nd4j.getExecutioner().execAndReturn(new IAMax(xKernel)).getFinalResult().intValue();
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
        return y;
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    @Override
    public INDArray update(INDArray gradient, double lr) {
        for (int ii = 0; ii < gradient.shape()[0]; ii++) {
            for (int fi = 0; fi < gradient.shape()[1]; fi++) {
                for (int vi = 0; vi < gradient.shape()[2]; vi++) {
                    for (int hi = 0; hi < gradient.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[]{
                                NDArrayIndex.point(ii), NDArrayIndex.point(fi),
                                NDArrayIndex.interval(vi * kernel[0], (vi + 1) * kernel[0]),
                                NDArrayIndex.interval(hi * kernel[1], (hi + 1) * kernel[1])
                        };
                        INDArray xKernel = mask.get(indexes).mul(gradient.getNumber(ii, fi, vi, hi));
                        mask.put(indexes, xKernel);
                    }
                }
            }
        }
        return mask;
    }

    /**
     * Display the layer on the standard output.
     */
    @Override
    public void print() {
        System.out.println("Type: MaxPooling2d");
        System.out.println();
    }
}
