package cnn.benchmark;

import cnn.NeuralNetwork;
import cnn.dataset.DataSet;
import cnn.dataset.DataSetFactory;
import cnn.layers.Conv2d;
import cnn.layers.Dense;
import cnn.layers.Flatten;
import cnn.layers.MaxPooling2d;
import cnn.layers.conf.ConfConv2d;

class CNN_vs_HCNN {

    public static void time(String type, NeuralNetwork net, DataSet dataSet) {
        // Training phase.
        double lr = 0.01;
        long start_time = System.nanoTime();
        net.fit(dataSet, lr, 1, 100);
        long end_time = System.nanoTime();
        double difference = (end_time - start_time) / 1e6;
        System.out.println("Training (" + type + "): " + difference + "ms");

        // Testing phase.
        net.evaluate(dataSet);
    }

    public static void main(String[] args) {
        try {
            // Create data set.
            DataSet dataSet = DataSetFactory.create("Mnist", 20);

            // Create neural networks.
            NeuralNetwork hcnn = new NeuralNetwork()
                    .addLayer(new Conv2d(new ConfConv2d(3, 0.01)))
                    .addLayer(new MaxPooling2d())
                    .addLayer(new Flatten())
                    .addLayer(new Dense(dataSet.getNumberOfClasses()));

            NeuralNetwork cnn = new NeuralNetwork()
                    .addLayer(new Conv2d())
                    .addLayer(new MaxPooling2d())
                    .addLayer(new Flatten())
                    .addLayer(new Dense(dataSet.getNumberOfClasses()));

            // Evaluation.
            time("HCNN", hcnn, dataSet);
            time("CNN", cnn, dataSet);

        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
