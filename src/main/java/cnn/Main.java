package cnn;

import cnn.dataset.DataSet;
import cnn.dataset.DataSetFactory;
import cnn.layers.Conv2d;
import cnn.layers.Dense;
import cnn.dataset.impl.MnistDataSet;
import cnn.layers.Flatten;
import cnn.layers.MaxPooling2d;
import cnn.layers.conf.ConfConv2d;

public class Main {

    /**
     * Entry point.
     * @param args program's arguments.
     */
    public static void main(String[] args) {
        // Create data set and neural networks.
        DataSet dataSet = DataSetFactory.create("Mnist", 20);

        NeuralNetwork network = new NeuralNetwork()
                .addLayer(new Conv2d(new ConfConv2d(3, 0.01)))
                .addLayer(new MaxPooling2d())
                .addLayer(new Flatten())
                .addLayer(new Dense(dataSet.getNumberOfClasses()));

        // Training phase.
        double lr = 0.01;
        network.fit(dataSet, lr, 1, 100);

        // Testing phase.
        network.evaluate(dataSet);

        // Inform me by email.
        //  MailHelper.sendTrainingIsOver();
    }
}
