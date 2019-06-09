package cnn.layers.perf;

import cnn.layers.conf.ConfConv2d;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Conv2dInterface {
    INDArray activation(ConfConv2d conf, INDArray x, INDArray w, INDArray bw);

    INDArray inputsGradients(ConfConv2d conf, long[] yShape, INDArray w, INDArray g);

    INDArray weightsGradients(ConfConv2d conf, long[] yShape, INDArray x, INDArray g);
}
