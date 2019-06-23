package cnn.ops;

import cnn.useful.ArrayPtr;

public interface ActivationInterface {

    enum Type {
        SIGMOID,
        RELU
    }

    ArrayPtr apply(Type af, ArrayPtr x);
    ArrayPtr derivative(Type af, ArrayPtr x);
}
