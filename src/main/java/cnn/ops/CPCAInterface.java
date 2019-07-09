package cnn.ops;

import cnn.nodes.conf.Conv2dConf;
import cnn.data.ArrayPtr;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface CPCAInterface {
    INDArray weightsGradients(Conv2dConf conf, ArrayPtr x, ArrayPtr w, ArrayPtr y);
}
