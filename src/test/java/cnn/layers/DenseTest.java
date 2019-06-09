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

    @Test
    void activationManyImages() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}, {2, 2}, {0, 1}});
        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w);
        INDArray y = layer.activation(x, false);
        assertEquals(6, y.getDouble(0, 0), "The first output must be equal to 6.");
        assertEquals(12, y.getDouble(0, 1), "The second output must be equal to 12.");
        assertEquals(4, y.getDouble(1, 0), "The thrid output must be equal to 4.");
        assertEquals(8, y.getDouble(1, 1), "The fourth output must be equal to 8.");
        assertEquals(1, y.getDouble(2, 0), "The fifth output must be equal to 1.");
        assertEquals(2, y.getDouble(2, 1), "The sixth output must be equal to 2.");
    }


    @Test
    void activationLargeImages() {
        INDArray w = arange2d(new long[]{11, 2});
        INDArray x = arange2d(new long[]{3, 10});
        INDArray t = Nd4j.create(new float[][]{{660, 706}, {1760, 1906}, {2860, 3106}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);

        for (int i = 0; i < t.shape()[0]; i++) {
            for (int j = 0; j < t.shape()[1]; j++) {
                assertEquals(t.getDouble(i, j), y.getDouble(i, j), "Check expected output is valid.");
            }
        }
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

    double computeNumericalGradientInput(Dense layer, int[] wp, INDArray x, INDArray t) {
        double epsilon = 0.01;
        double xVal = x.getDouble(wp);
        x.putScalar(wp, xVal - epsilon);
        INDArray ym = layer.activation(x, false).dup();
        x.putScalar(wp, xVal + epsilon);
        INDArray yp = layer.activation(x, false).dup();
        x.putScalar(wp, xVal);
        return (NeuralNetwork.SSE(t, yp) - NeuralNetwork.SSE(t, ym)) / (2 * epsilon);
    }

    INDArray arange2d(long[] shape) {
        INDArray res = Nd4j.create(shape);
        float n = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                res.putScalar(i, j, n);
                n += 1;
            }
        }
        return res;
    }

    @Test
    void gradientReluLargeImages() {
        INDArray w = arange2d(new long[]{11, 2});
        INDArray x = arange2d(new long[]{3, 10});
        INDArray t = arange2d(new long[]{3, 2});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        INDArray gradient = layer.getW().sub(w).mul(-1);
        layer.setW(w.dup());
        for (int i = 0; i < w.shape()[0]; i++) {
            for (int j = 0; j < w.shape()[1]; j++) {
                double grad = computeNumericalGradient(layer, new int[]{i, j}, x, t);
                assertTrue(Math.abs(grad - gradient.getDouble(i, j)) < grad / 100, "Numerical gradient checking."); // 1% tolerance
            }
        }
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
    void gradientReluWith2Samples() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}, {2, 1}});
        INDArray t = Nd4j.create(new float[][]{{1, 0}, {1, 0}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        INDArray gradient = layer.getW().sub(w).mul(-1);
        layer.setW(w.dup());
        for (int i = 0; i < w.shape()[0]; i++) {
            for (int j = 0; j < w.shape()[1]; j++) {
                double grad = computeNumericalGradient(layer, new int[]{i, j}, x, t);
                assertTrue(Math.abs(grad - gradient.getDouble(i, j)) < 0.5, "Numerical gradient checking.");
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

    @Test
    void gradientInput() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}});
        INDArray t = Nd4j.create(new float[][]{{1, 0}});

        Dense layer = new Dense(2, Activation.Type.SIGMOID);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);
        INDArray gradient = layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        layer.setW(w.dup());
        assertEquals(gradient.shape()[0], 1, "Check gradient shape.");
        assertEquals(gradient.shape()[1], 2, "Check gradient shape.");

        for (int i = 0; i < x.shape()[0]; i++) {
            for (int j = 0; j < x.shape()[1]; j++) {
                double grad = computeNumericalGradientInput(layer, new int[]{i, j}, x, t);
                assertTrue(Math.abs(grad - gradient.getDouble(i, j)) < 0.1, "Numerical gradient checking.");
            }
        }
    }

    @Test
    void gradientInputWith2Samples() {
        INDArray w = Nd4j.create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        INDArray x = Nd4j.create(new float[][]{{3, 3}, {2, 4}});
        INDArray t = Nd4j.create(new float[][]{{1, 0}, {0, 1}});

        Dense layer = new Dense(2, Activation.Type.SIGMOID);
        layer.setW(w.dup());
        INDArray y = layer.activation(x, true);
        INDArray gradient = layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        layer.setW(w.dup());

        assertEquals(gradient.shape()[0], 2, "Check gradient shape.");
        assertEquals(gradient.shape()[1], 2, "Check gradient shape.");

        for (int i = 0; i < x.shape()[0]; i++) {
            for (int j = 0; j < x.shape()[1]; j++) {
                double grad = computeNumericalGradientInput(layer, new int[]{i, j}, x, t);
                assertTrue(Math.abs(grad - gradient.getDouble(i, j)) < 0.1, "Numerical gradient checking.");
            }
        }
    }
}
