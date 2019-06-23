package cnn.useful.debug;

import cnn.NeuralNetwork;
import cnn.dataset.DataSet;

public interface DebugFuntion {
    void print(int i, NeuralNetwork nn, DataSet dataSet);
}
