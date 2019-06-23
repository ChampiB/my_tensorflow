package cnn.layers.impl.cpu;

import cnn.useful.ArrayPtr;
import org.junit.jupiter.api.Test;

import static cnn.TestHelper.*;

class MaxPooling2dTest {

    @Test
    void activation() {
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}});
        ArrayPtr t = create(new float[][][][]{{{{6, 8}, {14, 16}}}});

        MaxPooling2d layer = new MaxPooling2d(new int[]{2, 2});
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void update() {
        ArrayPtr x = create(new float[][][][]{{{{6, 2, 3, 8}, {5, 1, 7, 4}, {9, 10, 11, 12}, {14, 13, 15, 16}}}});
        ArrayPtr gradient = create(new float[][][][]{{{{1, 2}, {3, 4}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 0, 0, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {3, 0, 0, 4} }}});

        MaxPooling2d layer = new MaxPooling2d(new int[]{2, 2});
        layer.activation(x, true);
        gradient = layer.update(gradient, 0.01);
        assertEquals(t, gradient);
    }
}
