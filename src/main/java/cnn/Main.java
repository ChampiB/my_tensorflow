package cnn;

import cnn.layers.Conv2d;
import cnn.layers.Dense;
import cnn.dataset.MnistDataSet;
import cnn.layers.Flatten;
import cnn.layers.MaxPooling2d;

public class Main {

    /**
     * Entry point.
     * @param args program's arguments.
     */
    public static void main(String[] args) {
        // Create data set and neural networks.
        MnistDataSet mnist = new MnistDataSet(123, 20, 28, 28);

        int outputSize = 10;

        NeuralNetwork cnn = new NeuralNetwork()
                .addLayer(new Conv2d())
                .addLayer(new MaxPooling2d())
                .addLayer(new Flatten())
                .addLayer(new Dense(outputSize));

        // TODO NeuralNetwork hcnn = new NeuralNetwork()
        // TODO         .addLayer(new Conv2d(new ConfConv2d(3, 0.5)))
        // TODO         .addLayer(new MaxPooling2d())
        // TODO         .addLayer(new Flatten())
        // TODO         .addLayer(new Dense(outputSize));

        // Training phase.
        double lr = 0.01;
        cnn.fit(mnist, lr, 10, 100);
        // TODO hcnn.fit(mnist, lr, 1, 10);

        // Testing phase.
        cnn.evaluate(mnist);
        // TODO hcnn.evaluate(mnist);

        // Inform me by email.
        // TODO
        //  MailHelper.sendTrainingIsOver();
    }
}
