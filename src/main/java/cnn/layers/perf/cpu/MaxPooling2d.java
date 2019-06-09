package cnn.layers.perf.cpu;

import cnn.layers.perf.MaxPooling2dInterface;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MaxPooling2d implements MaxPooling2dInterface {
    /**
     * Compute max pooling.
     * @param kernel the kernel/pooling size.
     * @param x the input.
     * @param training true if training and false otherwise.
     * @return a pair containing the output and the mask if training is true.
     */
    public Pair<INDArray, INDArray> maxPooling2d(int[] kernel, INDArray x, boolean training) {
        long[] shape = x.shape();
        INDArray y = Nd4j.zeros(shape[0], shape[1], shape[2] / kernel[0], shape[3] / kernel[1]);
        INDArray mask = (training) ? Nd4j.zeros(shape) : Nd4j.zeros();
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
        return new ImmutablePair<>(y, mask);
    }

    /**
     * Compute the gradient with respect to the inputs.
     * @param kernel the kernel/pooling size.
     * @param g the gradient with respect to the output.
     * @param m the pooling mask.
     * @return the gradients.
     */
    public INDArray inputsGradients(int[] kernel, INDArray g, INDArray m) {
        for (int ii = 0; ii < g.shape()[0]; ii++) {
            for (int fi = 0; fi < g.shape()[1]; fi++) {
                for (int vi = 0; vi < g.shape()[2]; vi++) {
                    for (int hi = 0; hi < g.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[]{
                                NDArrayIndex.point(ii), NDArrayIndex.point(fi),
                                NDArrayIndex.interval(vi * kernel[0], (vi + 1) * kernel[0]),
                                NDArrayIndex.interval(hi * kernel[1], (hi + 1) * kernel[1])
                        };
                        INDArray xKernel = m.get(indexes).mul(g.getNumber(ii, fi, vi, hi));
                        m.put(indexes, xKernel);
                    }
                }
            }
        }
        return m;
    }
}
