package cnn.layers.perf.cpu;

import cnn.layers.conf.ConfConv2d;
import cnn.layers.perf.CPCAInterface;
import cnn.layers.perf.TasksFactory;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

class CPCATest {

    @Test
    void weightsGradients() {
        ConfConv2d conf = new ConfConv2d().setFilters(new int[]{1, 2, 2}).setStrides(new int[]{1, 1});
        INDArray w = Nd4j.create(new float[][][][]{{{{2, 2}, {2, 2}}}});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        INDArray y = Nd4j.create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        INDArray t = Nd4j.create(new float[][][][]{{{{1, 2}, {4, 5}}}});

        try {
            CPCAInterface cpca = TasksFactory.create("CPCA", "cpu");
            INDArray g = cpca.weightsGradients(conf, x, w, y);
            for (int i = 0; i < t.shape()[0]; i++) {
                for (int j = 0; j < t.shape()[1]; j++) {
                    for (int k = 0; k < t.shape()[1]; k++) {
                        assertEquals(t.getDouble(i, 0, j, k), g.getDouble(i, 0, j, k), "Gradient checking.");
                    }
                }
            }
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    void weightsGradientsWithZero() {
        ConfConv2d conf = new ConfConv2d().setFilters(new int[]{1, 2, 2}).setStrides(new int[]{1, 1});
        INDArray w = Nd4j.create(new float[][][][]{{{{2, 2}, {2, 2}}}});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        INDArray y = Nd4j.create(new float[][][][]{{{{1, 0}, {0, 1}}}});
        INDArray t = Nd4j.create(new float[][][][]{{{{1, 2}, {4, 5}}}});

        try {
            CPCAInterface cpca = TasksFactory.create("CPCA", "cpu");
            INDArray g = cpca.weightsGradients(conf, x, w, y);
            for (int i = 0; i < t.shape()[0]; i++) {
                for (int j = 0; j < t.shape()[1]; j++) {
                    for (int k = 0; k < t.shape()[1]; k++) {
                        assertEquals(t.getDouble(i, 0, j, k), g.getDouble(i, 0, j, k), "Gradient checking.");
                    }
                }
            }
        } catch (Exception e) {
            fail();
        }
    }
}
