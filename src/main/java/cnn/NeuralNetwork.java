package cnn;

import cnn.dataset.DataSet;
import cnn.layers.Layer;
import cnn.layers.LayersFactory;
import cnn.layers.conf.LayerConf;
import cnn.ops.OperationInterface;
import cnn.ops.OpsFactory;
import cnn.useful.ArrayPtr;
import cnn.useful.debug.DebugFuntion;
import cnn.useful.debug.MetricsDebugFunction;
import cnn.useful.stopping.EpochsStopCondition;
import cnn.useful.stopping.StopCondition;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.FileInputStream;
import java.io.FileOutputStream;
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
     * @param layer the name of the layer to add.
     * @return this.
     */
    public NeuralNetwork addLayer(String layer, LayerConf conf) {
        layers.add(LayersFactory.create(layer, conf));
        return this;
    }

    /**
     * Add a layer in the network.
     * @param layer the name of the layer to add.
     * @return this.
     */
    public NeuralNetwork addLayer(String layer) {
        layers.add(LayersFactory.create(layer));
        return this;
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
     * Getter.
     * @return an instance of OperationInterface.
     */
    private static OperationInterface op() {
        if (op == null) {
            op = OpsFactory.create("Operation");
        }
        return op;
    }
    private static OperationInterface op = null;

    /**
     * Compute the gradient of the SSE with respect to the output layer's activation.
     * @param t the target.
     * @param o the output.
     * @return the gradient of the SSE.
     */
    public static ArrayPtr SSEGradient(ArrayPtr t, ArrayPtr o) {
        op().sub(t, o);
        op().mul(t, -2);
        return t;
    }

    /**
     * Compute the SSE between the target and the output layer.
     * @param t the target.
     * @param o the output.
     * @return the gradient of the SSE.
     */
    public static double SSE(ArrayPtr t, ArrayPtr o) {
        t = t.dup();
        op().sub(t, o);
        op().pow(t, 2);
        return op().sum(t);
    }

    /**
     * Count the number of correct prediction.
     * @param t the target.
     * @param o the output.
     * @return the number of correct prediction.
     */
    public static double correctPredictions(ArrayPtr t, ArrayPtr o) {
        double correct = 0;
        for (int i = 0; i < t.getShape()[0]; i++)
            if (op().argMax(t, i) == op().argMax(o, i))
                correct += 1;
        return correct;
    }

    /**
     * Compute the network's activation.
     * @param x the input.
     * @param training the mode (training vs testing)
     * @return the activation.
     */
    public ArrayPtr activation(ArrayPtr x, boolean training) {
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
    private ArrayPtr update(ArrayPtr gradient, double lr) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).update(gradient, lr);
        }
        return gradient;
    }

    /**
     * Train the model on the data set.
     * @param dataSet the data set.
     * @param c the stopping condition.
     * @param lr the learning rate.
     * @param debug the debug function that can be used to display anything.
     */
    public void fit(DataSet dataSet, StopCondition c, double lr, DebugFuntion debug) {
        int i = 0;
        while (!c.shouldStop(i, this, dataSet)) {
            // Train the network.
            dataSet.nextBatch(true);
            ArrayPtr features = dataSet.getFeatures(true);
            ArrayPtr labels = dataSet.getLabels(true);
            ArrayPtr predictions = activation(features, true);
            ArrayPtr e = NeuralNetwork.SSEGradient(labels, predictions);
            update(e, lr);
            // Display debug if needed.
            debug.print(i, this, dataSet);
            i++;
        }
    }

    /**
     * Train the model on the data set.
     * @param dataSet the data set.
     * @param lr the learning rate.
     * @param epochs the number of training epochs.
     * @param debug the debug function that can be used to display anything.
     */
    public void fit(DataSet dataSet, double lr, int epochs, int debug) {
        fit(dataSet, new EpochsStopCondition(epochs), lr, new MetricsDebugFunction(debug));
    }

    /**
     * Evaluate the model on the data set.
     * @param dataSet the data set.
     */
    public void evaluate(DataSet dataSet) {
        // Ensure that the data set is loaded.
        if (!dataSet.hasNextBatch(false))
            dataSet.reload();
        // Evaluate the network.
        double totalSSE = 0;
        double totalCorrect = 0;
        double n = 0;
        while (dataSet.hasNextBatch(false)) {
            dataSet.nextBatch(false);
            ArrayPtr features = dataSet.getFeatures(false);
            ArrayPtr labels = dataSet.getLabels(false);
            ArrayPtr predictions = activation(features, false);
            totalSSE += NeuralNetwork.SSE(labels, predictions);
            totalCorrect += NeuralNetwork.correctPredictions(labels, predictions);
            n += labels.getShape()[0];
        }
        System.out.println("Mean SSE: " + (totalSSE / n));
        System.out.println("Accuracy: " + (totalCorrect / n));
        System.out.println();
    }

    /**
     * Print the network's layers.
     */
    public void printLayers() {
        for (Layer layer : layers) {
            layer.print();
        }
    }

    /**
     * Save the neural network into a file.
     * @param fileName the file's name.
     * @return true if saved correctly and false otherwise.
     */
    public boolean save(String fileName) {
        try {
            Kryo kryo = new Kryo();
            Output output = new Output(new FileOutputStream(fileName));
            kryo.writeObject(output, layers.size());
            for (Layer layer: layers) {
                layer.save(kryo, output);
            }
            output.close();
            return true;
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return false;
        }
    }

    /**
     * Load a neural network from the file.
     * @param fileName the file's name.
     * @return the neural network.
     */
    public NeuralNetwork loadWeights(String fileName) {
        try {
            Kryo kryo = new Kryo();
            Input input = new Input(new FileInputStream(fileName));
            int n = kryo.readObject(input, int.class);
            for (int i = 0; i < n; i++) {
                kryo.readObject(input, String.class);
                layers.get(i).loadWeights(kryo, input);
            }
            input.close();
            return this;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return null;
        }
    }

    /**
     * Load a neural network from the file.
     * @param fileName the file's name.
     * @return the neural network.
     */
    public static NeuralNetwork load(String fileName) {
        try {
            NeuralNetwork nn = new NeuralNetwork();
            Kryo kryo = new Kryo();
            Input input = new Input(new FileInputStream(fileName));
            int n = kryo.readObject(input, int.class);
            for (int i = 0; i < n; i++) {
                nn.addLayer(LayersFactory.create(kryo, input));
            }
            input.close();
            return nn;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return null;
        }
    }
}
