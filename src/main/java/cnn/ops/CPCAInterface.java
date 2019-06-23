package cnn.ops;

import cnn.layers.conf.Conv2dConf;
import cnn.useful.ArrayPtr;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface CPCAInterface {
    INDArray weightsGradients(Conv2dConf conf, ArrayPtr x, ArrayPtr w, ArrayPtr y);
}
