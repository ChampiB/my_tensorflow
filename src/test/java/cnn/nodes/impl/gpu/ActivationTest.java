package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;
import static cnn.nodes.enumerations.ActivationType.*;

class ActivationTest {
    @Test
    void activationNone() {
        ArrayPtr x = create(new float[][]{{3, 3}});
        ArrayPtr t = create(new float[][]{{3, 3}});

        Activation node = new Activation(NONE);
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void activationRelu() {
        ArrayPtr x = create(new float[][]{{3, -3}});
        ArrayPtr t = create(new float[][]{{3, 0}});

        Activation node = new Activation(RELU);
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void activationSoftMax() {
        ArrayPtr x = create(new float[][]{{3, 3, 3}});
        ArrayPtr t = create(new float[][]{{0.333f, 0.333f, 0.333f}});

        Activation node = new Activation(SOFTMAX);
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y, 1.0);
    }

    @Test
    void activationSigmoid() {
        ArrayPtr x = create(new float[][]{{-0.5f, 0f, 0.5f}});
        ArrayPtr t = create(new float[][]{{0.3775f, 0.5f, 0.6224f}});

        Activation node = new Activation(SIGMOID);
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y, 1.0);
    }

    @Test
    void updateNone() {
        ArrayPtr x = create(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        ArrayPtr t = create(new float[][]{{3, 2, 1}, {6, 5, 4}, {9, 8, 7}});

        Activation node = new Activation(NONE);
        ArrayPtr gradient = computeGradient_i(node, x, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1, 0.001f);
    }

    @Test
    void updateRelu() {
        ArrayPtr x = create(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        ArrayPtr t = create(new float[][]{{3, 2, 1}, {6, 5, 4}, {9, 8, 7}});

        Activation node = new Activation(RELU);
        ArrayPtr gradient = computeGradient_i(node, x, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1, 0.001f);
    }

    @Test
    void updateSoftMax() {
        ArrayPtr x = create(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        ArrayPtr t = create(new float[][]{{3, 2, 1}, {6, 5, 4}, {9, 8, 7}});

        Activation node = new Activation(SOFTMAX);
        ArrayPtr gradient = computeGradient_i(node, x, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1, 0.001f);
    }

    @Test
    void updateSigmoid() {
        ArrayPtr x = create(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        ArrayPtr t = create(new float[][]{{3, 2, 1}, {6, 5, 4}, {9, 8, 7}});

        Activation node = new Activation(SIGMOID);
        ArrayPtr gradient = computeGradient_i(node, x, t);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1, 0.001f);
    }
}
