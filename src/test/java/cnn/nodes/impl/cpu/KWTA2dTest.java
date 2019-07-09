package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;

class KWTA2dTest {

    @Test
    void activation() {
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}}, {{4, 5, 6}}, {{7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{0, 0, 0}}, {{0, 0, 0}}, {{7, 8, 9}}}});

        KWTA2d node = new KWTA2d(1);
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void updateValidOutput() {
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}}, {{4, 5, 6}}, {{7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{0, 0, 0}}, {{0, 0, 0}}, {{7, 8, 9}}}});

        KWTA2d layer = new KWTA2d(1);
        ArrayPtr gradient = computeGradient_i(layer, x, t);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void updateInvalidOutput() {
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}}, {{4, 5, 6}}, {{7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{0, 0, 0}}, {{0, 0, 0}}, {{1, 1, 1}}}});

        KWTA2d layer = new KWTA2d(1);
        ArrayPtr gradient = computeGradient_i(layer, t, x);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }
}