package cnn.graphs.conf;

import cnn.nodes.Node;
import cnn.nodes.NodesFactory;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.security.InvalidParameterException;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class GraphConf {

    private Map<Integer, Node> outputs = new LinkedHashMap<>();

    /**
     * Setter.
     * @param nodes the outputs of the graph.
     * @return this.
     */
    public GraphConf setOutputs(Node... nodes) {
        outputs.clear();
        for (int i = 0; i < nodes.length; i++) {
            outputs.put(i, nodes[i]);
        }
        if (isCyclic()) {
            throw new InvalidParameterException();
        }
        return this;
    }

    /**
     * Getter.
     * @param id the output's id.
     * @return the output node.
     */
    public Node getOutput(int id) {
        return outputs.get(id);
    }

    /**
     * Check if the graph is cyclic.
     *
     * @param path     the path being explored in the graph.
     * @param children the children of the current node.
     * @return true if cyclic and false otherwise.
     */
    private boolean isCyclic(List<Node> path, Node[] children) {
        for (Node child : children) {
            if (path.contains(child))
                return true;
            if (child == null)
                continue;
            path.add(child);
            boolean cyclic = isCyclic(path, child.getInputs());
            path.remove(path.size() - 1);
            if (cyclic)
                return true;
        }
        return false;
    }

    /**
     * Check if the graph is cyclic.
     *
     * @return true if cyclic and false otherwise.
     */
    public boolean isCyclic() {
        List<Node> path = new LinkedList<>();

        Node output = getOutput(0);
        path.add(output);
        Node[] children = output.getInputs();
        return isCyclic(path, children);
    }

    /**
     * Save the graph in the file.
     * @param fileName the file name.
     * @return true if success and false otherwise.
     */
    public boolean save(String fileName) {
        try {
            Kryo kryo = new Kryo();
            Output output = new Output(new FileOutputStream(fileName));
            save(kryo, output);
            output.close();
            return true;
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return false;
        }
    }

    /**
     * Save the configuration.
     * @param kryo the kryo object.
     * @param output the output.
     */
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, outputs.size());
        for (Map.Entry<Integer, Node> entry: outputs.entrySet()) {
            kryo.writeObject(output, entry.getKey());
            entry.getValue().save(kryo, output);
        }
    }

    /**
     * Load the weights of the configuration, i.e. do nothing except reading on the input.
     * @param fileName the file name
     * @return the configuration or null in cse of failure.
     */
    public GraphConf loadWeights(String fileName) {
        try {
            Kryo kryo = new Kryo();
            Input input = new Input(new FileInputStream(fileName));
            loadWeights(kryo, input);
            input.close();
            return this;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return null;
        }
    }

    /**
     * Load the weights of the configuration, i.e. do nothing except reading on the input.
     * @param kryo the kryo configuration.
     * @param input the input.
     * @return this.
     */
    public GraphConf loadWeights(Kryo kryo, Input input) {
        // Load outputs.
        int outputsSize = kryo.readObject(input, Integer.class);
        for (int i = 0; i < outputsSize; i++) {
            int outputId = kryo.readObject(input, Integer.class);
            kryo.readObject(input, String.class);
            outputs.get(outputId).loadWeights(kryo, input);
        }
        return this;
    }

    /**
     * Load the configuration from the file.
     * @param fileName the file's name.
     * @return this or null in case of failure.
     */
    public static GraphConf load(String fileName) {
        try {
            Kryo kryo = new Kryo();
            Input input = new Input(new FileInputStream(fileName));
            GraphConf conf = load(kryo, input);
            input.close();
            return conf;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return null;
        }
    }

    /**
     * Load the configuration.
     * @param kryo the kryo object.
     * @param input the input.
     * @return the configuration.
     */
    public static GraphConf load(Kryo kryo, Input input) {
        GraphConf conf = new GraphConf();
        List<Node> outputs = new LinkedList<>();
        // Load outputs.
        int outputsSize = kryo.readObject(input, Integer.class);
        for (int i = 0; i < outputsSize; i++) {
            kryo.readObject(input, Integer.class);
            outputs.add(NodesFactory.create(kryo, input));
        }
        conf.setOutputs( (Node[]) outputs.toArray() );
        return conf;
    }

    /**
     * Print the graph configuration.
     */
    public void print() {
        System.out.println("Number of output: " + outputs.size() + ".");
    }
}
