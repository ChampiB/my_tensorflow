package cnn.nodes.impl.cpu.useful;

import cnn.nodes.conf.Conv2dConf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.concurrent.Callable;

public class ConvolutionTask implements Callable<Boolean> {

    private INDArray input;
    private INDArray output;
    private int index;
    private int[] filters;
    private int[] strides;
    private INDArray w;
    private INDArray bw;

    /**
     * Constructor.
     * @param input the activation of the previous layer.
     * @param output the output (i.e. after the convolution).
     * @param index the example's index that must be processed.
     * @param conf the layer's configuration.
     * @param w the layer's weights
     * @param bw  the layer's bias weights
     */
    public ConvolutionTask(INDArray input, INDArray output, int index, Conv2dConf conf, INDArray w, INDArray bw) {
        this.input = input;
        this.output = output;
        this.index = index;
        this.filters = conf.filters();
        this.strides = conf.strides();
        this.w = w;
        this.bw = bw;
    }

    /**
     * Apply the convolution on the index-th example.
     * @return true if the task success and false otherwise.
     */
    @Override
    public Boolean call() {
        int voffset = 0;
        for (int vi = 0; vi < output.shape()[2]; vi++) {
            int hoffset = 0;
            for (int hi = 0; hi < output.shape()[3]; hi++) {
                INDArray kernel = input.slice(index).get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(voffset, voffset + filters[1]),
                        NDArrayIndex.interval(hoffset, hoffset + filters[2])
                );
                for (int fi = 0; fi < filters[0]; fi++) {
                    output.putScalar(
                            index, fi, vi, hi,
                            kernel.mul(w.slice(fi)).sumNumber().doubleValue() + bw.getNumber(fi).doubleValue()
                    );
                }
                hoffset += strides[1];
            }
            voffset += strides[0];
        }
        return true;
    }
}
