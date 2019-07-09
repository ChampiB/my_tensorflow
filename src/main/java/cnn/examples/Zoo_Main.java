package cnn.examples;

import cnn.dataset.DataSet;
import cnn.dataset.DataSetsFactory;
import cnn.graphs.Graph;
import cnn.zoo.ModelFactory;

public class Zoo_Main {

    /**
     * Entry point.
     * @param args program's arguments.
     */
    public static void main(String[] args) {
        // Create data set.
        int batchSize = (args.length < 3) ? 20 : Integer.valueOf(args[2]);
        String dName = (args.length < 2) ? "Mnist" : args[1];
        DataSet dataSet = DataSetsFactory.create(dName, batchSize);

        // Create neural graphs.
        String mName = (args.length < 1) ? "AlexNet" : args[0];
        Graph network = ModelFactory.create(mName, 10);

        // Training phase.
        double lr = 0.001;
        network.fit(dataSet, lr, 10, 100);

        // Testing phase.
        network.evaluate(dataSet);
    }
}
