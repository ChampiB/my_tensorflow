package cnn.useful.debug;

import cnn.NeuralNetwork;
import cnn.dataset.DataSet;
import cnn.useful.ArrayPtr;

public class AccuracyTestingSetDebugFunction implements DebugFuntion {

    private int debug;

    public AccuracyTestingSetDebugFunction(int debug) {
        this.debug = debug;
    }

    @Override
    public void print(int i, NeuralNetwork nn, DataSet dataSet) {
        if (i == 0)
            System.out.println("iteration,accuracy");
        if (i % debug == 0) {
            double totalCorrect = 0;
            double n = 0;
            while (dataSet.hasNextBatch(false)) {
                dataSet.nextBatch(false);
                ArrayPtr features = dataSet.getFeatures(false);
                ArrayPtr labels = dataSet.getLabels(false);
                ArrayPtr predictions = nn.activation(features, false);
                totalCorrect += NeuralNetwork.correctPredictions(labels, predictions);
                n += labels.getShape()[0];
            }
            System.out.println(i + "," + (totalCorrect / n));
            dataSet.reload(false);
        }
    }
}
