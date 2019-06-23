package cnn.useful.stopping;

import cnn.NeuralNetwork;
import cnn.dataset.DataSet;

public interface StopCondition {
    boolean shouldStop(int i, NeuralNetwork nn, DataSet dataSet);
}
