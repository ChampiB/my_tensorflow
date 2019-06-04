package cnn.perf.gpu;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.CPCAInterface;
import cnn.perf.TasksFactory;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

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
            CPCAInterface cpca = TasksFactory.create("CPCA", "gpu");
            INDArray g = cpca.weightsGradients(conf, x, w, y);
            for (int i = 0; i < t.shape()[0]; i++) {
                for (int j = 0; j < t.shape()[1]; j++) {
                    for (int k = 0; k < t.shape()[1]; k++) {
                        assertEquals(t.getDouble(i, 0, j, k), g.getDouble(i, 0, j, k), "Gradient checking.");
                    }
                }
            }
        } catch (TasksFactory.NoGPUException e) {
            System.err.println(e.getMessage());
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
            CPCAInterface cpca = TasksFactory.create("CPCA", "gpu");
            INDArray g = cpca.weightsGradients(conf, x, w, y);
            for (int i = 0; i < t.shape()[0]; i++) {
                for (int j = 0; j < t.shape()[1]; j++) {
                    for (int k = 0; k < t.shape()[1]; k++) {
                        assertEquals(t.getDouble(i, 0, j, k), g.getDouble(i, 0, j, k), "Gradient checking.");
                    }
                }
            }
        } catch (TasksFactory.NoGPUException e) {
            System.err.println(e.getMessage());
        }
    }
}
