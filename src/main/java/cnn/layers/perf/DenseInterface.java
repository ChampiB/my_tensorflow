package cnn.layers.perf;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface DenseInterface {
    INDArray activation(INDArray x, INDArray w);

    INDArray inputsGradients(long[] yShape, INDArray w, INDArray g);

    INDArray weightsGradients(long[] yShape, INDArray x, INDArray g);
}
