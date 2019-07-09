package cnn.nodes.impl.cpu;

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
        assertEquals(x1, y[0]);
        assertEquals(x2, y[1]);
        assertEquals(x3, y[2]);
    }
}