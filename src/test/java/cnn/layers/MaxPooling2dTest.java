package cnn.layers;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class MaxPooling2dTest {

    @Test
    void activation() {
        INDArray x = Nd4j.create(new float[][][][]{{{
            {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}
        }}});
        MaxPooling2d layer = new MaxPooling2d(new int[]{2, 2});
        INDArray y = layer.activation(x, false);
        assertEquals(6, y.getDouble(0, 0, 0, 0));
        assertEquals(8, y.getDouble(0, 0, 0, 1));
        assertEquals(14, y.getDouble(0, 0, 1, 0));
        assertEquals(16, y.getDouble(0, 0, 1, 1));
    }

    @Test
    void update() {
        INDArray x = Nd4j.create(new float[][][][]{{{
                {6, 2, 3, 8},
                {5, 1, 7, 4},
                {9, 10, 11, 12},
                {14, 13, 15, 16}
        }}});
        MaxPooling2d layer = new MaxPooling2d(new int[]{2, 2});
        layer.activation(x, true);
        INDArray gradient = Nd4j.create(new float[][][][]{{{
                {1, 2}, {3, 4}
        }}});
        gradient = layer.update(gradient, 0.01);
        INDArray t = Nd4j.create(new float[][][][]{{{
                {1, 0, 0, 2},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {3, 0, 0, 4}
        }}});
        for (int vi = 0; vi < t.shape()[2]; vi++) {
            for (int hi = 0; hi < t.shape()[3]; hi++) {
                assertEquals(t.getDouble(0, 0, vi, hi), gradient.getDouble(0, 0, vi, hi));
            }
        }
    }
}