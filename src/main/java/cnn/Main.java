package cnn;

import cnn.layers.Conv2d;
import cnn.layers.Dense;
import cnn.dataset.MnistDataSet;
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
        MnistDataSet mnist = new MnistDataSet(20, 28, 28);

        int outputSize = 10;

        // NeuralNetwork cnn = new NeuralNetwork()
        //         .addLayer(new Conv2d())
        //         .addLayer(new MaxPooling2d())
        //         .addLayer(new Flatten())
        //         .addLayer(new Dense(outputSize));

        NeuralNetwork hcnn = new NeuralNetwork()
                .addLayer(new Conv2d(new ConfConv2d(3, 0.01)))
                .addLayer(new MaxPooling2d())
                .addLayer(new Flatten())
                .addLayer(new Dense(outputSize));

        // Training phase.
        double lr = 0.01;
        // cnn.fit(mnist, lr, 10, 100);
        hcnn.fit(mnist, lr, 1, 100);

        // Testing phase.
        // cnn.evaluate(mnist);
        hcnn.evaluate(mnist);

        // Inform me by email.
        //  MailHelper.sendTrainingIsOver();
    }
}
