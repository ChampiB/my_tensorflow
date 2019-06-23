package cnn;

import cnn.dataset.DataSet;
import cnn.dataset.DataSetsFactory;
import cnn.layers.conf.Conv2dConf;
import cnn.layers.conf.DenseConf;

public class Main {

    /**
     * Entry point.
     * @param args program's arguments.
     */
    public static void main(String[] args) {
        // The network's filename.
        String file = "network.save";

        // Create data set.
        DataSet dataSet = DataSetsFactory.create("Mnist", 20);

        // Create neural networks.
        NeuralNetwork network = NeuralNetwork.load(file);
        if (network == null) {
            network = new NeuralNetwork()
                    .addLayer("Conv2d", new Conv2dConf(3, 0.01))
                    .addLayer("MaxPooling2d")
                    .addLayer("Flatten")
                    .addLayer("Dense", new DenseConf(100))
                    .addLayer("Dense", new DenseConf(dataSet.getNumberOfClasses()));
        }

        // Training phase.
        double lr = 0.001;
        network.fit(dataSet, lr, 1, 100);

        // Testing phase.
        network.evaluate(dataSet);

        // Save network.
        network.save(file);
    }
}
