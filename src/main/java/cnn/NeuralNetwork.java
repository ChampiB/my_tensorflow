package cnn;

import cnn.dataset.MnistDataSet;
import cnn.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * Neural network.
 */
public class NeuralNetwork {

    private List<Layer> layers;

    /**
     * Create a neural network.
     */
    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    /**
     * Add a layer in the network.
     * @param layer the layer to add.
     * @return this.
     */
    public NeuralNetwork addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Compute the gradient of the SSE with respect to the output layer's activation.
     * @param t the target.
     * @param o the output.
     * @return the gradient of the SSE.
     */
    public static INDArray SSEGradient(INDArray t, INDArray o) {
        return t.sub(o).mul(-2);
    }

    /**
     * Compute the SSE between the target and the output layer.
     * @param t the target.
     * @param o the output.
     * @return the gradient of the SSE.
     */
    public static double SSE(INDArray t, INDArray o) {
        return Transforms.pow(t.sub(o), 2).sumNumber().doubleValue();
    }

    /**
     * Count the number of correct prediction.
     * @param t the target.
     * @param o the output.
     * @return the number of correct prediction.
     */
    private double correctPredictions(INDArray t, INDArray o) {
        double correct = 0;
        for (int i = 0; i < t.shape()[0]; i++) {
            if (t.getRow(i).argMax().getDouble(0) == o.getRow(i).argMax().getDouble(0))
                correct += 1;
        }
        return correct;
    }

    /**
     * Compute the network's activation.
     * @param x the input.
     * @param training the mode (training vs testing)
     * @return the activation.
     */
    private INDArray activation(INDArray x, boolean training) {
        for (Layer layer : layers) {
            x = layer.activation(x, training);
        }
        return x;
    }

    /**
     * Update the layers' weights.
     * @param gradient the gradient of the output layer.
     * @param lr the learning rate.
     */
    private INDArray update(INDArray gradient, double lr) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).update(gradient, lr);
        }
        return gradient;
    }

    /**
     * Train the model on the data set.
     * @param dataSet the data set.
     * @param lr the learning rate.
     */
    public void fit(MnistDataSet dataSet, double lr) {
        fit(dataSet, lr, 100);
    }

    /**
     * Train the model on the data set.
     * @param dataSet the data set.
     * @param lr the learning rate.
     * @param debug the number of iteration between each display of metrics.
     */
    public void fit(MnistDataSet dataSet, double lr, int debug) {
        // Ensure that the data set is loaded.
        if (!dataSet.hasNextBatch(true))
            dataSet.reload();
        // Train the network.
        int i = 0;
        while (dataSet.hasNextBatch(true)) {
            DataSet trainingSet = dataSet.nextBatch(true);
            INDArray features = trainingSet.getFeatures().reshape(dataSet.batchShape());
            INDArray labels = trainingSet.getLabels();
            INDArray predictions = activation(features, true);
            INDArray e = NeuralNetwork.SSEGradient(labels, predictions);
            update(e, lr);
            if (i % debug == 0) {
                double n = labels.shape()[0];
                System.out.println("Mean SSE/" + i + ": " + (SSE(labels, predictions) / n));
                System.out.println("Accuracy/" + i + ": " + (correctPredictions(labels, predictions) / n));
                System.out.println();
            }
            i++;
        }
    }

    /**
     * Train the model on the data set.
     * @param dataSet the data set.
     * @param lr the learning rate.
     * @param debug the number of iteration between each display of metrics.
     */
    public void fit(MnistDataSet dataSet, double lr, int epochs, int debug) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epochs:" + (i + 1) + "/" + epochs + ".");
            fit(dataSet, lr, debug);
            dataSet.reload();
        }
    }

    /**
     * Evaluate the model on the data set.
     * @param dataSet the data set.
     */
    public void evaluate(MnistDataSet dataSet) {
        double totalSSE = 0;
        double totalCorrect = 0;
        double n = 0;
        while (dataSet.hasNextBatch(false)) {
            DataSet trainingSet = dataSet.nextBatch(false);
            INDArray features = trainingSet.getFeatures().reshape(dataSet.batchShape());
            INDArray labels = trainingSet.getLabels();
            INDArray predictions = activation(features, false);
            totalSSE += SSE(labels, predictions);
            totalCorrect += correctPredictions(labels, predictions);
            n += labels.shape()[0];
        }
        System.out.println("Mean SSE: " + (totalSSE / n));
        System.out.println("Accuracy: " + (totalCorrect / n));
        System.out.println();
    }

    /**
     * Print the network's layers.
     */
    public void printLayers() {
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).print();
        }
    }
}
