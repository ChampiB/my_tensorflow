package cnn.useful.debug;

import cnn.graphs.Graph;
import cnn.dataset.DataSet;

public interface DebugFunction {
    void print(int i, Graph nn, DataSet dataSet);
}
