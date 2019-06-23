package cnn.useful.debug;

import cnn.NeuralNetwork;
import cnn.dataset.DataSet;
import cnn.useful.ArrayPtr;

public class MetricsDebugFunction implements DebugFuntion {

    private int debug;

    public MetricsDebugFunction(int debug) {
        this.debug = debug;
    }

    @Override
    public void print(int i, NeuralNetwork nn, DataSet dataSet) {
        if (i % debug == 0) {
            ArrayPtr x = dataSet.getFeatures(true);
            ArrayPtr t = dataSet.getLabels(true);
            ArrayPtr y = nn.activation(x, true);
            double n = t.getShape()[0];
            System.out.println("Mean SSE/" + i + ": " + (NeuralNetwork.SSE(t, y) / n));
            System.out.println("Accuracy/" + i + ": " + (NeuralNetwork.correctPredictions(t, y) / n));
            System.out.println();
        }
    }
}
