package cnn.useful.stopping;

import cnn.graphs.Graph;
import cnn.dataset.DataSet;

public interface StopCondition {
    boolean shouldStop(int i, Graph nn, DataSet dataSet);
}
