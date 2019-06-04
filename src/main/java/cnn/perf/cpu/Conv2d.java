package cnn.perf.cpu;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.Conv2dInterface;
import cnn.perf.cpu.useful.ConvolutionTask;
import cnn.perf.cpu.useful.ThreadPool;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Future;

public class Conv2d implements Conv2dInterface {
    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param x the input.
     * @param w the weights.
     * @param bw the bias weights.
     * @return the output.
     */
    public INDArray conv2d(ConfConv2d conf, INDArray x, INDArray w, INDArray bw) {
        // Compute the number of vertical and horizontal position.
        long nr = x.shape()[2] - conf.filters()[1] + 1;
        nr = (long) Math.ceil(((double)nr) / ((double)conf.strides()[0]));
        long nc = x.shape()[3] - conf.filters()[2] + 1;
        nc = (long) Math.ceil(((double)nc) / ((double)conf.strides()[1]));
        // Launch one convolution task for each image.
        INDArray result = Nd4j.zeros(x.shape()[0], conf.filters()[0], nr, nc);
        List<Future<Boolean>> tasks = new LinkedList<>();
        for (int ii = 0; ii < x.shape()[0]; ii++) {
            tasks.add(ThreadPool.getInstance().submit(new ConvolutionTask(x, result, ii, conf, w, bw)));
        }
        ThreadPool.waitAll(tasks);
        return result;
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray inputsGradients(ConfConv2d conf, long[] yShape, INDArray w, INDArray g) {
        INDArray di = Nd4j.zeros(yShape);
        for (int fi = 0; fi < g.shape()[1]; fi++) {
            for (int ii = 0; ii < g.shape()[0]; ii++) {
                INDArray weights = w.slice(fi);
                int voff = 0;
                for (int vi = 0; vi < g.shape()[2]; vi++) {
                    int hoff = 0;
                    for (int hi = 0; hi < g.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[] {
                                NDArrayIndex.point(ii),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                                NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                        };
                        INDArray gi = weights.mul(g.getDouble(ii, fi, vi, hi));
                        di.put(indexes, di.get(indexes).add(gi));
                        hoff += conf.strides()[1];
                    }
                    voff += conf.strides()[0];
                }
            }
        }
        return di;
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray weightsGradients(ConfConv2d conf, long[] yShape, INDArray x, INDArray g) {
        INDArray dw = Nd4j.zeros(yShape);
        for (int fi = 0; fi < g.shape()[1]; fi++) {
            for (int ii = 0; ii < g.shape()[0]; ii++) {
                int voff = 0;
                for (int vi = 0; vi < g.shape()[2]; vi++) {
                    int hoff = 0;
                    for (int hi = 0; hi < g.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[] {
                                NDArrayIndex.point(ii),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                                NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                        };
                        INDArray gw = x.get(indexes).mul(g.getDouble(ii, fi, vi, hi));
                        dw.putSlice(fi, dw.slice(fi).add(gw));
                        hoff += conf.strides()[1];
                    }
                    voff += conf.strides()[0];
                }
            }
        }
        return dw;
    }
}
