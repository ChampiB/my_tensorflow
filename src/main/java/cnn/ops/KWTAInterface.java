package cnn.ops;

import cnn.layers.conf.Conv2dConf;
import cnn.useful.ArrayPtr;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface KWTAInterface {
    ArrayPtr activation(Conv2dConf conf, ArrayPtr x);
}
