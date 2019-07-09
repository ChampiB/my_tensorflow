package cnn.nodes;

import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.util.LinkedList;
import java.util.List;

/**
 * Generic layer abstraction.
 */
public abstract class Node {

    private Node[] inputs = new Node[]{null};
    private List<Node> outputs = new LinkedList<>();

    /**
     * Compute the layer activation.
     * @param training the mode (training vs testing).
     * @param x is the input.
     * @return the activation.
     */
    public abstract ArrayPtr activation(boolean training, ArrayPtr... x);

    /**
     * Update the weights.
     * @param lr the learning rate.
     * @param gradient the back propagation gradient from the upper layer.
     * @return the back propagation gradient from this layer.
     */
    public abstract ArrayPtr[] update(double lr, ArrayPtr... gradient);

    /**
     * Save the layer to the file.
     * @param kryo the kryo object.
     * @param output the kryo output.
     */
    public abstract void save(Kryo kryo, Output output);

    /**
     * Load weights from file.
     * @param kryo the kryo object.
     * @param input the kryo input.
     */
    public abstract Node loadWeights(Kryo kryo, Input input);

    /**
     * Load layer from file.
     * @param kryo the kryo object.
     * @param input the kryo input.
     */
    public abstract Node load(Kryo kryo, Input input);

    /**
     * Display the layer on the standard output.
     */
    public abstract void print();

    /**
     * Setter.
     * @param inputs of the node.
     * @return this.
     */
    public Node setInputs(Node... inputs) {
        for (Node input: inputs)
            input.addOutput(this);
        this.inputs = inputs;
        return this;
    }

    /**
     * Getter.
     * @return the inputs of the node.
     */
    public Node[] getInputs() {
        return inputs;
    }

    /**
     * Add an output to the node.
     * @param output the new output.
     * @return this.
     */
    private Node addOutput(Node output) {
        outputs.add(output);
        return this;
    }

    /**
     * Getter.
     * @return the outputs of the node.
     */
    public List<Node> getOutputs() {
        return outputs;
    }
}
