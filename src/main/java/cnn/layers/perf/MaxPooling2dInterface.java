package cnn.layers.perf;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface MaxPooling2dInterface {
    /**
     * Compute max pooling.
     * @param kernel the kernel/pooling size.
     * @param x the input.
     * @param training true if training and false otherwise.
     * @return a pair containing the output and the mask if training is true.
     */
    Pair<INDArray, INDArray> maxPooling2d(int[] kernel, INDArray x, boolean training);

    /**
     * Compute the gradient with respect to the inputs.
     * @param kernel the kernel/pooling size.
     * @param g the gradient with respect to the output.
     * @param mask the pooling mask.
     * @return the gradients.
     */
    INDArray inputsGradients(int[] kernel, INDArray g, INDArray mask);
}
