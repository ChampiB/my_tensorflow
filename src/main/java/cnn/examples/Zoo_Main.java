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
        DataSet dataSet = DataSetsFactory.create("Mnist", 20);

        // Create neural graphs.
        String name = (args.length == 0) ? "AlexNet" : args[0];
        Graph network = ModelFactory.create(name, 10);

        // Training phase.
        double lr = 0.001;
        network.fit(dataSet, lr, 1, 100);

        // Testing phase.
        network.evaluate(dataSet);
    }
}
