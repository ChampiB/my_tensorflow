package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;

class AddTest {

    @Test
    void activationOneInput() {
        ArrayPtr x = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{1, 1}});

        Add node = new Add();
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void activationTwoInputs() {
        ArrayPtr x1 = create(new float[][]{{1, 1}});
        ArrayPtr x2 = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{2, 2}});

        Add node = new Add();
        ArrayPtr y = node.activation(false, x1, x2);
        assertEquals(t, y);
    }

    @Test
    void activationThreeInputs() {
        ArrayPtr x1 = create(new float[][]{{1, 1}});
        ArrayPtr x2 = create(new float[][]{{1, 1}});
        ArrayPtr x3 = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{3, 3}});

        Add node = new Add();
        ArrayPtr y = node.activation(false, x1, x2, x3);
        assertEquals(t, y);
    }

    @Test
    void updateOneInput() {
        ArrayPtr x = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{3, 3}});
        Add node = new Add();

        ArrayPtr[] gradient = computeGradient_i(node, t, x);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x)[0]);
        assertEquals(tg, gradient[0], 1);
    }

    @Test
    void updateTwoInputs() {
        ArrayPtr x1 = create(new float[][]{{1, 1}});
        ArrayPtr x2 = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{3, 3}});
        Add node = new Add();

        ArrayPtr[] gradients = computeGradient_i(node, t, x1, x2);
        INDArray tg0 = generate2d(x1.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x1, x2)[0]);
        assertEquals(tg0, gradients[0], 1);
        INDArray tg1 = generate2d(x1.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x1, x2)[1]);
        assertEquals(tg1, gradients[1], 1);
    }

    @Test
    void updateThreeInputs() {
        ArrayPtr x1 = create(new float[][]{{1, 1}});
        ArrayPtr x2 = create(new float[][]{{1, 1}});
        ArrayPtr x3 = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{3, 3}});
        Add node = new Add();

        ArrayPtr[] gradients = computeGradient_i(node, t, x1, x2, x3);
        INDArray tg0 = generate2d(x1.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x1, x2, x3)[0]);
        assertEquals(tg0, gradients[0], 1);
        INDArray tg1 = generate2d(x1.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x1, x2, x3)[1]);
        assertEquals(tg1, gradients[1], 1);
        INDArray tg2 = generate2d(x1.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x1, x2, x3)[2]);
        assertEquals(tg2, gradients[1], 1);
    }
}