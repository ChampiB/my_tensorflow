package cnn.useful.debug.impl;

import cnn.graphs.Graph;
import cnn.dataset.DataSet;
import cnn.data.ArrayPtr;
import cnn.useful.debug.DebugFunction;

public class AccuracyTrainingSet implements DebugFunction {

    private int debug;

    public AccuracyTrainingSet(int debug) {
        this.debug = debug;
    }

    @Override
    public void print(int i, Graph nn, DataSet dataSet) {
        if (i == 0)
            System.out.println("iteration,accuracy");
        if (i % debug == 0) {
            ArrayPtr x = dataSet.getFeatures(true);
            ArrayPtr t = dataSet.getLabels(true);
            ArrayPtr y = nn.activation(true, x);
            System.out.println(i + "," + (Graph.correctPredictions(t, y) / t.getShape()[0]));
        }
    }
}
