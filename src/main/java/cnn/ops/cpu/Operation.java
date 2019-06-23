package cnn.ops.cpu;

import cnn.ops.OperationInterface;
import cnn.useful.ArrayPtr;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Operation implements OperationInterface {

    public Operation() {}

    @Override
    public int argMax(ArrayPtr a, int row) {
        return a.toCPU().getRow(row).argMax().getInt();
    }

    @Override
    public float sum(ArrayPtr a) {
        return a.toCPU().sumNumber().floatValue();
    }

    @Override
    public void sub(ArrayPtr a1, ArrayPtr a2) {
        a1.copy(a1.toCPU().sub(a2.toCPU()));
    }

    @Override
    public void pow(ArrayPtr a1, int n) {
        a1.copy(Transforms.pow(a1.toCPU(), n));

    }

    @Override
    public void mul(ArrayPtr a1, ArrayPtr a2) {
        a1.copy(a1.toCPU().mul(a2.toCPU()));
    }

    @Override
    public void mul(ArrayPtr a1, float n) {
        a1.copy(a1.toCPU().mul(n));
    }

    @Override
    public void add(ArrayPtr a1, ArrayPtr a2) {
        a1.copy(a1.toCPU().add(a2.toCPU()));
    }
}
