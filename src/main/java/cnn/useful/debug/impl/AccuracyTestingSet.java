package cnn.useful.debug.impl;

import cnn.graphs.Graph;
import cnn.dataset.DataSet;
import cnn.data.ArrayPtr;
import cnn.useful.debug.DebugFunction;

public class AccuracyTestingSet implements DebugFunction {

    private int debug;

    public AccuracyTestingSet(int debug) {
        this.debug = debug;
    }

    @Override
    public void print(int i, Graph nn, DataSet dataSet) {
        if (i == 0)
            System.out.println("iteration,accuracy");
        if (i % debug == 0) {
            double totalCorrect = 0;
            double n = 0;
            while (dataSet.hasNextBatch(false)) {
                dataSet.nextBatch(false);
                ArrayPtr features = dataSet.getFeatures(false);
                ArrayPtr labels = dataSet.getLabels(false);
                ArrayPtr predictions = nn.activation(false, features);
                totalCorrect += Graph.correctPredictions(labels, predictions);
                n += labels.getShape()[0];
            }
            System.out.println(i + "," + (totalCorrect / n));
            dataSet.reload(false);
        }
    }
}
