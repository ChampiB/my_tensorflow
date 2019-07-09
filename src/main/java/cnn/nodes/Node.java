package cnn.nodes;

import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Generic layer abstraction.
 */
public abstract class Node {

    private List<Node> inputs = new LinkedList<>(Arrays.asList(new Node[]{null}));
    private List<Node> outputs = new LinkedList<>();

    /**
     * Compute the node activation.
     * @param training the mode (training vs testing).
     * @param x is the input.
     * @return the activation.
     */
    public abstract ArrayPtr activation(boolean training, ArrayPtr... x);

    /**
     * Update the weights.
     * @param lr the learning rate.
     * @param gradients the back propagation gradients from the upper nodes.
     * @return the gradients with respect to the node's inputs.
     */
    public abstract ArrayPtr[] update(double lr, ArrayPtr... gradients);

    /**
     * Save the node to the file.
     * @param kryo the kryo object.
     * @param output the kryo output.
     */
    public abstract void save(Kryo kryo, Output output);

    /**
     * Load weights from file.
     * @param kryo the kryo object.
     * @param input the kryo input.
     * @return this.
     */
    public abstract Node loadWeights(Kryo kryo, Input input);

    /**
     * Load node from file.
     * @param kryo the kryo object.
     * @param input the kryo input.
     * @return this.
     */
    public abstract Node load(Kryo kryo, Input input);

    /**
     * Display the node on the standard output.
     */
    public abstract void print();

    /**
     * Setter.
     * @param inputs of the node.
     * @return this.
     */
    public Node setInputs(Node... inputs) {
        for (Node input: inputs)
            if (input != null)
                input.addOutput(this);
        this.inputs = new LinkedList<>(Arrays.asList(inputs));
        return this;
    }

    /**
     * Setter.
     * @param inputs of the node.
     * @return this.
     */
    public Node addInputs(Node... inputs) {
        for (Node input: inputs) {
            if (input != null)
                input.addOutput(this);
            this.inputs.add(input);
        }
        return this;
    }
    /**
     * Getter.
     * @return the inputs of the node.
     */
    public Node[] getInputs() {
        return inputs.toArray(new Node[0]);
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
