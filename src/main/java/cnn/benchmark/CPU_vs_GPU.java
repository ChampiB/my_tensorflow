package cnn.benchmark;

import cnn.dataset.DataSet;
import cnn.dataset.DataSetFactory;
import cnn.layers.Conv2d;
import cnn.layers.Dense;
import cnn.layers.Flatten;
import cnn.layers.MaxPooling2d;
import cnn.layers.conf.ConfConv2d;
import cnn.layers.perf.TasksFactory;

import cnn.NeuralNetwork;

class CPU_vs_GPU {

    public static void time(String type) {
        // Create data set and neural networks.
        DataSet dataSet = DataSetFactory.create("Mnist", 20);

        TasksFactory.forceImplementation(type);
        NeuralNetwork hcnn = new NeuralNetwork()
                .addLayer(new Conv2d(new ConfConv2d(3, 0.01)))
                .addLayer(new MaxPooling2d())
                .addLayer(new Flatten())
                .addLayer(new Dense(dataSet.getNumberOfClasses()));

        // Training phase.
        double lr = 0.01;
        long start_time = System.nanoTime();
        hcnn.fit(dataSet, lr, 1, 100);
        long end_time = System.nanoTime();
        double difference = (end_time - start_time) / 1e6;
        System.out.println("Training (" + type + "): " + difference + "ms");

        // Testing phase.
        hcnn.evaluate(dataSet);
    }

    public static void main(String[] args) {
        try {
            time("gpu");
            time("cpu");
        } catch (Exception e) {
            System.err.println("No GPU available.");
        }
    }
}
