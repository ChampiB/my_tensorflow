package cnn.nodes.impl.cpu;

import cnn.nodes.conf.PadConf;
import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;

class Pad2dTest {

    @Test
    void activationDimZero() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}}}, {{{0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(0, 2, 0));
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void activationDimOne() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}}, {{0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(1, 2, 0));
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void activationDimTwo() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}, {0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(2, 2, 0));
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void activationDimThree() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1, 0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(3, 4, 0));
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void updateDimZero() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}}}, {{{0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(0, 2, 0));
        ArrayPtr gradient = computeGradient_i(node, t, x);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void updateDimOne() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}}, {{0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(1, 2, 0));
        ArrayPtr gradient = computeGradient_i(node, t, x);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void updateDimTwo() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}, {0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(2, 2, 0));
        ArrayPtr gradient = computeGradient_i(node, t, x);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void updateDimThree() {
        ArrayPtr x = create(new float[][][][]{{{{1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1, 0, 0}}}});

        Pad2d node = new Pad2d(new PadConf(3, 4, 0));
        ArrayPtr gradient = computeGradient_i(node, t, x);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, x, t));
        assertEquals(tg, gradient, 1);
    }
}