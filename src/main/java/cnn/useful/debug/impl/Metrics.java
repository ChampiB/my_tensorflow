package cnn.useful.debug.impl;

import cnn.graphs.Graph;
import cnn.dataset.DataSet;
import cnn.data.ArrayPtr;
import cnn.useful.debug.DebugFunction;

public class Metrics implements DebugFunction {

    private int debug;

    public Metrics(int debug) {
        this.debug = debug;
    }

    @Override
    public void print(int i, Graph nn, DataSet dataSet) {
        if (i % debug == 0) {
            ArrayPtr x = dataSet.getFeatures(true);
            ArrayPtr t = dataSet.getLabels(true);
            ArrayPtr y = nn.activation(true, x);
            double n = t.getShape()[0];
            System.out.println("Mean SSE/" + i + ": " + (Graph.SSE(t, y) / n));
            System.out.println("Accuracy/" + i + ": " + (Graph.correctPredictions(t, y) / n));
            System.out.println();
        }
    }
}
