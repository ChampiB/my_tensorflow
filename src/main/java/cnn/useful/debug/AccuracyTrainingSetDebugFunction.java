package cnn.useful.debug;

import cnn.NeuralNetwork;
import cnn.dataset.DataSet;
import cnn.useful.ArrayPtr;

public class AccuracyTrainingSetDebugFunction implements DebugFuntion {

    private int debug;

    public AccuracyTrainingSetDebugFunction(int debug) {
        this.debug = debug;
    }

    @Override
    public void print(int i, NeuralNetwork nn, DataSet dataSet) {
        if (i == 0)
            System.out.println("iteration,accuracy");
        if (i % debug == 0) {
            ArrayPtr x = dataSet.getFeatures(true);
            ArrayPtr t = dataSet.getLabels(true);
            ArrayPtr y = nn.activation(x, true);
            System.out.println(i + "," + (NeuralNetwork.correctPredictions(t, y) / t.getShape()[0]));
        }
    }
}
