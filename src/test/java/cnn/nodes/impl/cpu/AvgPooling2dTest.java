package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;
import static cnn.helpers.TestHelper.numericalGradient_i;

class AvgPooling2dTest {

    @Test
    void activation() {
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}});
        ArrayPtr t = create(new float[][][][]{{{{3.5f, 5.5f}, {11.5f, 13.5f}}}});

        AvgPooling2d layer = new AvgPooling2d(new int[]{2, 2});
        ArrayPtr y = layer.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void update() {
        ArrayPtr x = create(new float[][][][]{{{{6, 2, 3, 8}, {5, 1, 7, 4}, {9, 10, 11, 12}, {14, 13, 15, 16}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 2}, {3, 4}}}});

        AvgPooling2d layer = new AvgPooling2d(new int[]{2, 2});
        ArrayPtr gradient = computeGradient_i(layer, t, x);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }
}