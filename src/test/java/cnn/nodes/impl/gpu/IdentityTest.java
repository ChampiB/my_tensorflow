package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.helpers.TestHelper.*;
import static cnn.helpers.TestHelper.assertEquals;

class IdentityTest {

    @Test
    void activation() {
        ArrayPtr x = create(new float[][]{{1.42f, 3.14f}});
        ArrayPtr t = create(new float[][]{{1.42f, 3.14f}});

        Identity node = new Identity();
        ArrayPtr y = node.activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void update() {
        ArrayPtr x = create(new float[][]{{1.42f, 3.14f}});
        ArrayPtr t = create(new float[][]{{1.42f, 3.14f}});

        Identity node = new Identity();
        ArrayPtr gradient = computeGradient_i(node, t, x);
        INDArray tg = generate2d(x.getShape(), (pos) -> (float)numericalGradient_i(node, pos, t, x));
        assertEquals(tg, gradient, 1);
    }
}