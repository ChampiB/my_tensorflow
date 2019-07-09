package cnn.graphs.impl;

import cnn.nodes.Node;
import cnn.nodes.NodesFactory;
import cnn.graphs.Graph;

/**
 * Neural network.
 */
public class NeuralNetwork extends Graph {

    private Node prev = null;

    /**
     * Create a neural network.
     */
    public NeuralNetwork() {}

    /**
     * Add a layer in the network.
     * @param layer the name of the layer to add.
     * @return this.
     */
    public NeuralNetwork addLayer(String layer, Object conf) {
        return addLayer(NodesFactory.create(layer, conf));
    }

    /**
     * Add a layer in the network.
     * @param layer the name of the layer to add.
     * @return this.
     */
    public NeuralNetwork addLayer(String layer) {
        return addLayer(NodesFactory.create(layer));
    }

    /**
     * Add a node in the network.
     * @param node the node to add.
     * @return this.
     */
    public NeuralNetwork addLayer(Node node) {
        node.setInputs(prev);
        getConf().setOutputs(node);
        prev = node;
        return this;
    }
}
