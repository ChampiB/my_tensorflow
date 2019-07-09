package cnn.ops.gpu;

import cnn.nodes.conf.Conv2dConf;
import cnn.ops.CPCAInterface;
import cnn.data.ArrayPtr;
import cnn.ops.OpsFactory;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;

class CPCATest {

    @Test
    void weightsGradients() {
        Conv2dConf conf = new Conv2dConf().setFilters(new int[]{1, 2, 2}).setStrides(new int[]{1, 1});
        ArrayPtr w = create(new float[][][][]{{{{2, 2}, {2, 2}}}});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr y = create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 2}, {4, 5}}}});

        CPCAInterface cpca = OpsFactory.create("CPCA", "gpu");
        INDArray g = cpca.weightsGradients(conf, x, w, y);
        assertEquals(t, g);
    }

    @Test
    void weightsGradientsWithZero() {
        Conv2dConf conf = new Conv2dConf().setFilters(new int[]{1, 2, 2}).setStrides(new int[]{1, 1});
        ArrayPtr w = create(new float[][][][]{{{{2, 2}, {2, 2}}}});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr y = create(new float[][][][]{{{{1, 0}, {0, 1}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 2}, {4, 5}}}});

        CPCAInterface cpca = OpsFactory.create("CPCA", "gpu");
        INDArray g = cpca.weightsGradients(conf, x, w, y);
        assertEquals(t, g);
    }
}
