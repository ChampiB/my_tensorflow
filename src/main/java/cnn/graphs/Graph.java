package cnn.graphs;

import cnn.dataset.DataSet;
import cnn.nodes.Node;
import cnn.graphs.conf.GraphConf;
import cnn.graphs.impl.NeuralNetwork;
import cnn.ops.OperationInterface;
import cnn.ops.OpsFactory;
import cnn.data.ArrayPtr;
import cnn.useful.debug.DebugFunction;
import cnn.useful.debug.impl.Metrics;
import cnn.useful.stopping.EpochsStopCondition;
import cnn.useful.stopping.StopCondition;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.security.InvalidParameterException;
import java.util.*;

/**
 * Generic graph of computation.
 */
public class Graph extends Node {

    private Map<Node, ArrayPtr[]> gradientsMap = new LinkedHashMap<>();
    private GraphConf conf;
    private ArrayPtr inputGradients;

    /**
     * Constructor.
     * @param conf the graph configuration.
     */
    public Graph(GraphConf conf) {
        setConf(conf);
    }

    /**
     * Default constructor.
     */
    public Graph() {
        this.conf = new GraphConf();
    }

    /**
     * Getter.
     * @return the graph configuration.
     */
    public GraphConf getConf() {
        return conf;
    }

    /**
     * Setter.
     * @param conf the configuration.
     */
    public void setConf(GraphConf conf) {
        if (conf.isCyclic()) {
            throw new InvalidParameterException();
        }
        this.conf = conf;
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
     * Train the model on the data set.
     * @param dataSet the data set.
     * @param c the stopping condition.
     * @param lr the learning rate.
     * @param debug the debug function that can be used to display anything.
     */
    public void fit(DataSet dataSet, StopCondition c, double lr, DebugFunction debug) {
        int i = 0;
        while (!c.shouldStop(i, this, dataSet)) {
            // Train the network.
            dataSet.nextBatch(true);
            ArrayPtr features = dataSet.getFeatures(true);
            ArrayPtr labels = dataSet.getLabels(true);
            ArrayPtr predictions = activation(true, features);
            ArrayPtr e = NeuralNetwork.SSEGradient(labels, predictions);
            update(lr, e);
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
        fit(dataSet, new EpochsStopCondition(epochs), lr, new Metrics(debug));
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
            ArrayPtr predictions = activation(false, features);
            totalSSE += NeuralNetwork.SSE(labels, predictions);
            totalCorrect += NeuralNetwork.correctPredictions(labels, predictions);
            n += labels.getShape()[0];
        }
        System.out.println("Mean SSE: " + (totalSSE / n));
        System.out.println("Accuracy: " + (totalCorrect / n));
        System.out.println();
    }

    /**
     * Save the neural network into a file.
     * @param fileName the file's name.
     * @return true if saved correctly and false otherwise.
     */
    public boolean save(String fileName) {
        return conf.save(fileName);
    }

    /**
     * Load a neural network from the file.
     * @param fileName the file's name.
     * @return the neural network.
     */
    public Graph loadWeights(String fileName) {
        conf.loadWeights(fileName);
        return this;
    }

    /**
     * Load a neural network from the file.
     * @param fileName the file's name.
     * @return the neural network.
     */
    public static Graph load(String fileName) {
        GraphConf conf = GraphConf.load(fileName);
        return (conf == null) ? null : new Graph(conf);
    }

    /**
     * Comppute the activation of the graph.
     * @param x the inputs.
     * @param training true if training mode and false otherwise.
     * @param path the path being explored in the graph.
     * @param children the children of the current node.
     * @return the activation.
     */
    private ArrayPtr activation(ArrayPtr x, boolean training, List<Node> path, Node[] children) {
        ArrayPtr[] inputs = new ArrayPtr[children.length];

        for (int i = 0; i < children.length; i++) {
            Node child = children[i];
            path.add(child);
            inputs[i] = (child == null) ? x : activation(x, training, path, child.getInputs());
            path.remove(path.size() - 1);
        }
        return path.get(path.size() - 1).activation(training, inputs);
    }

    /**
     * Comppute the activation of the graph.
     * @param x the inputs.
     * @param training true if training mode and false otherwise.
     * @return the activation.
     */
    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        List<Node> path = new LinkedList<>();

        Node output = conf.getOutput(0);
        path.add(output);
        return activation(x[0], training, path, output.getInputs());
    }

    /**
     * Create or fill an entry in the gradient map.
     * @param node the key of the entry.
     * @param nbOutputs the number of outputs.
     * @param gradients the gradients with respect to the node output.
     */
    private void addGradientsToMapEntry(Node node, int nbOutputs, ArrayPtr... gradients) {
        ArrayPtr[] gi = gradientsMap.getOrDefault(node, null);
        if (gi == null) {
            ArrayPtr[] grad = new ArrayPtr[nbOutputs];
            for (int i = 0; i < nbOutputs; i++) {
                grad[i] = (i < gradients.length) ? gradients[i] : null;
            }
            gradientsMap.put(node, grad);
        } else {
            for (ArrayPtr gradient: gradients) {
                for (int i = 0; i < gi.length; i++) {
                    if (gi[i] == null) {
                        gi[i] = gradient;
                        break;
                    }
                }
            }
        }
    }

    /**
     * Free the input gradient.
     */
    private void freeInputGradient() {
        if (inputGradients != null)
            inputGradients.freeGpu();
        inputGradients = null;
    }

    /**
     * Check if an imput is missing.
     * @param inputs the inputs.
     * @return true if an input is missing (i.e. null) and false otherwise.
     */
    private boolean isInputMissing(ArrayPtr[] inputs) {
        for (ArrayPtr input : inputs) {
            if (input == null) {
                return true;
            }
        }
        return false;
    }

    /**
     * Sum all the inputs.
     * @param inputs the inputs.
     * @return the sum.
     */
    private ArrayPtr sumAll(ArrayPtr[] inputs) {
        ArrayPtr sum = inputs[0].dup();
        for (int i = 1; i < inputs.length; i++) {
            op().add(sum, inputs[i].dup());
        }
        return sum;
    }

    /**
     * Update the graph.
     * @param lr the learning rate.
     * @param path the path being explored in the graph.
     * @param children the children of the current node.
     */
    public void update(double lr, List<Node> path, Node[] children) {
        Node last = path.get(path.size() - 1);
        if (isInputMissing(gradientsMap.get(last))) {
            return;
        }

        ArrayPtr gradient = sumAll(gradientsMap.get(last));
        ArrayPtr[] gradients = last.update(lr, gradient);

        for (int i = 0; i < children.length; i++) {
            if (children[i] == null) {
                if (inputGradients == null) {
                    inputGradients = gradients[i].dup();
                } else {
                    op().add(inputGradients, gradients[i]);
                }
                continue;
            }
            addGradientsToMapEntry(children[i], children[i].getOutputs().size(), gradients[i]);
            path.add(children[i]);
            update(lr, path, children[i].getInputs());
            path.remove(path.size() - 1);
        }
    }

    /**
     * Update the graph.
     * @param lr the learning rate.
     * @param gradients the gradient with respect to the graph output.
     * @return the gradient with respect to the graph input.
     */
    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradients) {
        List<Node> path = new LinkedList<>();

        freeInputGradient();
        Node output = conf.getOutput(0);
        path.add(output);
        addGradientsToMapEntry(output, 1, gradients);
        update(lr, path, output.getInputs());
        gradientsMap.clear();
        return new ArrayPtr[]{inputGradients.dup()};
    }

    /**
     * Save the graph.
     * @param kryo the kryo object.
     * @param output the kryo output.
     */
    @Override
    public void save(Kryo kryo, Output output) {
        conf.save(kryo, output);
    }

    /**
     * Load the weights.
     * @param kryo the kryo object.
     * @param input the kryo input.
     * @return this.
     */
    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        conf.loadWeights(kryo, input);
        return this;
    }

    /**
     * Load the graph.
     * @param kryo the kryo object.
     * @param input the kryo input.
     * @return this.
     */
    @Override
    public Node load(Kryo kryo, Input input) {
        conf = GraphConf.load(kryo, input);
        return this;
    }

    /**
     * Print the graph.
     */
    @Override
    public void print() {
        System.out.println("Type: Graph");
        conf.print();
    }
}
