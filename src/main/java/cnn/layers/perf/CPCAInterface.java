package cnn.layers.perf;

import cnn.layers.conf.ConfConv2d;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface CPCAInterface {
    INDArray weightsGradients(ConfConv2d conf, INDArray x, INDArray w, INDArray y);
}
