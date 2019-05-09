package cnn.layers;

import cnn.layers.activation.Activation;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import cnn.NeuralNetwork;

import static org.junit.jupiter.api.Assertions.*;

class DenseTest {

    @Test
    void activation() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}});
        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w);
        INDArray y = layer.activation(x, false);
        assertEquals(6, y.getDouble(0, 0), "The first output must be equal to 6.");
        assertEquals(12, y.getDouble(0, 1), "The second output must be equal to 12.");
    }

    double computeNumericalGradient(Dense layer, int[] wp, INDArray x, INDArray t) {
        double epsilon = 0.01;
        double w = layer.getW().getDouble(wp);
        layer.getW().putScalar(wp, w - epsilon);
        INDArray ym = layer.activation(x, false).dup();
        layer.getW().putScalar(wp, w + epsilon);
        INDArray yp = layer.activation(x, false).dup();
        layer.getW().putScalar(wp, w);
        return (NeuralNetwork.SSE(t, yp) - NeuralNetwork.SSE(t, ym)) / (2 * epsilon);
    }

    @Test
    void gradientRelu() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}});
        INDArray t = Nd4j.create(new float[][]{{1, 0}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        INDArray gradient = layer.getW().sub(w).mul(-1);
        layer.setW(w.dup());

        for (int i = 0; i < w.shape()[0]; i++) {
            for (int j = 0; j < w.shape()[1]; j++) {
                double grad = computeNumericalGradient(layer, new int[]{i, j}, x, t);
                assertTrue(Math.abs(grad - gradient.getDouble(i, j)) < 0.01, "Numerical gradient checking.");
            }
        }
    }

    @Test
    void gradientSigmoid() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}});
        INDArray t = Nd4j.create(new float[][]{{1, 0}});

        Dense layer = new Dense(2, Activation.Type.SIGMOID);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        INDArray gradient = layer.getW().sub(w).mul(-1);
        layer.setW(w.dup());

        for (int i = 0; i < w.shape()[0]; i++) {
            for (int j = 0; j < w.shape()[1]; j++) {
                double grad = computeNumericalGradient(layer, new int[]{i, j}, x, t);
                assertTrue(Math.abs(grad - gradient.getDouble(i, j)) < 0.1, "Numerical gradient checking.");
            }
        }
    }
}