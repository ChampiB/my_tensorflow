package cnn.perf;

import cnn.layers.conf.ConfConv2d;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface KWTAInterface {
    INDArray kwta(ConfConv2d conf, INDArray x);
}
