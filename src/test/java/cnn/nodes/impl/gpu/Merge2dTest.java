package cnn.nodes.impl.gpu;

import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;

import static cnn.helpers.TestHelper.assertEquals;
import static cnn.helpers.TestHelper.create;

class Merge2dTest {

    @Test
    void activation() {
        ArrayPtr x1 = create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        ArrayPtr x2 = create(new float[][][][]{{{{2, 2}, {2, 2}}}});
        ArrayPtr x3 = create(new float[][][][]{{{{3, 3}, {3, 3}}}});
        ArrayPtr t = create(new float[][][][]{{
                {{1, 1}, {1, 1}},
                {{2, 2}, {2, 2}},
                {{3, 3}, {3, 3}}
        }});

        Merge2d node = new Merge2d();
        ArrayPtr y = node.activation(false, x1, x2, x3);
        assertEquals(t, y);
    }

    @Test
    void update() {
System.out.println("ACT start");
        ArrayPtr x1 = create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        ArrayPtr x2 = create(new float[][][][]{{{{2, 2}, {2, 2}}}});
        ArrayPtr x3 = create(new float[][][][]{{{{3, 3}, {3, 3}}}});
        ArrayPtr t = create(new float[][][][]{{
                {{1, 1}, {1, 1}},
                {{2, 2}, {2, 2}},
                {{3, 3}, {3, 3}}
        }});

        Merge2d node = new Merge2d();
        node.activation(true, x1, x2, x3);
        ArrayPtr[] y = node.update(1, t);
System.out.println("x1:" + x1.toCPU());
System.out.println("y1:" + y[0].toCPU());
System.out.println("x2:" + x2.toCPU());
System.out.println("y2:" + y[1].toCPU());
System.out.println("x3:" + x3.toCPU());
System.out.println("y3:" + y[2].toCPU());
        assertEquals(x1, y[0]);
        assertEquals(x2, y[1]);
        assertEquals(x3, y[2]);
System.out.println("ACT end");
    }
}
