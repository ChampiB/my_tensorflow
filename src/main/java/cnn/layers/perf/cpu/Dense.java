package cnn.layers.perf.cpu;

import cnn.layers.perf.DenseInterface;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Dense implements DenseInterface {
    /**
     * Remove the first column corresponding to the bais input.
     * @param a the array.
     * @param ishape the input shape.
     * @return the new array without the bias.
     */
    private INDArray removeBiasColumn(INDArray a, long[] ishape) {
        long[] offset = new long[]{(int) ishape[0], 0};
        int[] shape = new int[]{(int) ishape[0], (int) ishape[1] - 1};
        int[] stride = new int[]{1, (int) ishape[0]};
        return a.subArray(offset, shape, stride);
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param x the input.
     * @param w the weights.
     * @return the output.
     */
    public INDArray activation(INDArray x, INDArray w) {
        return x.mmul(w);
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray inputsGradients(long[] yShape, INDArray w, INDArray g) {
        return removeBiasColumn(g.mmul(w.transpose()), yShape);
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param yShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray weightsGradients(long[] yShape, INDArray x, INDArray g) {
        return x.transpose().mmul(g);
    }
}
