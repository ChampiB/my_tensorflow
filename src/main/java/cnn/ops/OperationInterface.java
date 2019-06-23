package cnn.ops;

import cnn.useful.ArrayPtr;

public interface OperationInterface {
    int argMax(ArrayPtr a, int row);
    float sum(ArrayPtr a);
    void sub(ArrayPtr a1, ArrayPtr a2);
    void pow(ArrayPtr a1, int n);
    void mul(ArrayPtr a1, ArrayPtr a2);
    void mul(ArrayPtr a1, float n);
    void add(ArrayPtr a1, ArrayPtr a2);
}
