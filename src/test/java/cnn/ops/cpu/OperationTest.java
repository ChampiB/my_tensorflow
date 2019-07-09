package cnn.ops.cpu;

import cnn.helpers.TestHelper;
import cnn.ops.OperationInterface;
import cnn.ops.OpsFactory;
import cnn.data.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static cnn.helpers.TestHelper.create;

class OperationTest {

    @Test
    void argMax() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr x = create(new float[][]{{1, 2}, {2, 1}});

        assertEquals(1, op.argMax(x, 0));
        assertEquals(0, op.argMax(x, 1));
    }

    @Test
    void sum() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr x = create(new float[][]{{1, 2}, {2, 1}});

        assertEquals(6, op.sum(x));
    }

    @Test
    void sub() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr a1 = create(new float[][]{{1, 2}, {2, 1}});
        ArrayPtr a2 = create(new float[][]{{1, 2}, {2, 1}});
        INDArray t = Nd4j.zeros(a1.getShape());

        op.sub(a1, a2);
        TestHelper.assertEquals(t, a1);
    }

    @Test
    void pow() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr a1 = create(new float[][]{{1, 2}, {2, 1}});
        INDArray t = Transforms.pow(a1.toCPU().dup(), 2);

        op.pow(a1, 2);
        TestHelper.assertEquals(t, a1);
    }

    @Test
    void mulArray() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr a1 = create(new float[][]{{2, 1}, {1, 2}});
        ArrayPtr a2 = create(new float[][]{{1, 2}, {2, 1}});
        INDArray t = Nd4j.ones(a1.getShape()).mul(2);

        op.mul(a1, a2);
        TestHelper.assertEquals(t, a1);
    }

    @Test
    void mulScalar() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr a1 = create(new float[][]{{1, 2}, {2, 1}});
        ArrayPtr t = create(new float[][]{{10, 20}, {20, 10}});

        op.mul(a1, 10);
        TestHelper.assertEquals(t, a1);
    }

    @Test
    void add() {
        OperationInterface op = OpsFactory.create("Operation", "cpu");
        ArrayPtr a1 = create(new float[][]{{1, 2}, {2, 1}});
        ArrayPtr a2 = create(new float[][]{{1, 2}, {2, 1}});
        INDArray t = a2.toCPU().dup().mul(2);

        op.add(a1, a2);
        TestHelper.assertEquals(t, a1);
    }
}
