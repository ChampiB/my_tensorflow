package cnn.layers.impl.cpu;

import cnn.ops.cpu.Activation;
import cnn.useful.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.TestHelper.*;

class DenseTest {

    @Test
    void activation() {
        ArrayPtr w = create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        ArrayPtr x = create(new float[][]{{3, 3}});
        ArrayPtr t = create(new float[][]{{6, 12}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void activationManyImages() {
        ArrayPtr w = create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        ArrayPtr x = create(new float[][]{{3, 3}, {2, 2}, {0, 1}});
        ArrayPtr t = create(new float[][]{{6, 12}, {4, 8}, {1, 2}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void activationLargeImages() {
        ArrayPtr w = arrange(new long[]{11, 2});
        ArrayPtr x = arrange(new long[]{3, 10});
        ArrayPtr t = create(new float[][]{{660, 706}, {1760, 1906}, {2860, 3106}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        assertEquals(t, y);
    }

    @Test
    void gradientReluLargeImages() {
        ArrayPtr w = arrange(new long[]{11, 2});
        ArrayPtr x = arrange(new long[]{3, 10});
        ArrayPtr t = arrange(new long[]{3, 2});

        Dense layer = new Dense(2, Activation.Type.RELU);
        ArrayPtr gradient = computeGradient_w(layer, x, w, t);
        INDArray tg = generate2d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientRelu() {
        ArrayPtr w = create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        ArrayPtr x = create(new float[][]{{3, 3}});
        ArrayPtr t = create(new float[][]{{1, 0}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        ArrayPtr gradient = computeGradient_w(layer, x, w, t);
        INDArray tg = generate2d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientReluWith2Samples() {
        ArrayPtr w = create(new float[][]{{0, 0}, {1, 2}, {1, 2}});
        ArrayPtr x = create(new float[][]{{3, 3}, {2, 1}});
        ArrayPtr t = create(new float[][]{{1, 0}, {1, 0}});

        Dense layer = new Dense(2, Activation.Type.RELU);
        ArrayPtr gradient = computeGradient_w(layer, x, w, t);
        INDArray tg = generate2d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientSigmoid() {
        ArrayPtr w = create(new float[][]{{0, 0}, {10, 20}, {10, 20}});
        ArrayPtr x = create(new float[][]{{30, 30}});
        ArrayPtr t = create(new float[][]{{10, 0}});

        Dense layer = new Dense(2, Activation.Type.SIGMOID);
        ArrayPtr gradient = computeGradient_w(layer, x, w, t);
        INDArray tg = generate2d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientInput() {
        ArrayPtr w = create(new float[][]{{0, 0}, {10, 20}, {10, 20}});
        ArrayPtr x = create(new float[][]{{30, 30}});
        ArrayPtr t = create(new float[][]{{10, 0}});

        Dense layer = new Dense(2, Activation.Type.SIGMOID);
        ArrayPtr gradient = computeGradient_i(layer, x, w, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientInputWith2Samples() {
        ArrayPtr w = create(new float[][]{{0, 0}, {10, 20}, {10, 20}});
        ArrayPtr x = create(new float[][]{{30, 30}, {20, 40}});
        ArrayPtr t = create(new float[][]{{10, 0}, {0, 10}});

        Dense layer = new Dense(2, Activation.Type.SIGMOID);
        ArrayPtr gradient = computeGradient_i(layer, x, w, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientInputLargeImages() {
        ArrayPtr w = arrange(new long[]{11, 2});
        ArrayPtr x = arrange(new long[]{3, 10});
        ArrayPtr t = arrange(new long[]{3, 2});

        Dense layer = new Dense(2, Activation.Type.RELU);
        ArrayPtr gradient = computeGradient_i(layer, x, w, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }
}
