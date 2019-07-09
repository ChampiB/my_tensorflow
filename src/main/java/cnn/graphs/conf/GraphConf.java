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
     * Should add all the inputs of the node into the list.
     * Should not add the inputs that are already in the list.
     * @param node the node.
     * @param nodes the list.
     */
    private void addInputsRec(Node node, List<Node> nodes) {
        Node[] children = node.getInputs();
        for (Node child : children) {
            if (child != null && !nodes.contains(child)) {
                nodes.add(child);
                addInputsRec(child, nodes);
            }
        }
    }

    /**
     * Count the number of links.
     * @param nodes the nodes of the graph.
     * @return the number of links.
     */
    private int countLinks(List<Node> nodes) {
        int nbLinks = 0;
        for (Node node : nodes) {
            for (Node input : node.getInputs()) {
                if (input != null) nbLinks++;
            }
        }
        return nbLinks;
    }

    /**
     * Get list of all nodes.
     * This function should insert all the output at the beginning of the list.
     * @return the nodes of the graph.
     */
    private List<Node> getAllNodes() {
        List<Node> nodes = new LinkedList<>();
        for (Map.Entry<Integer, Node> entry: outputs.entrySet()) {
            nodes.add(entry.getValue());
        }
        for (Map.Entry<Integer, Node> entry: outputs.entrySet()) {
            addInputsRec(entry.getValue(), nodes);
        }
        return nodes;
    }

    /**
     * Save the configuration.
     * @param kryo the kryo object.
     * @param output the output.
     */
    public void save(Kryo kryo, Output output) {
        // Save the number of outputs and outputs' id.
        kryo.writeObject(output, outputs.size());
        for (Map.Entry<Integer, Node> entry: outputs.entrySet()) {
            kryo.writeObject(output, entry.getKey());
        }
        // Save the number of nodes and all nodes.
        List<Node> nodes = getAllNodes();
        kryo.writeObject(output, nodes.size());
        for (Node node : nodes) {
            node.save(kryo, output);
        }
        // Save the number of links and all links.
        kryo.writeObject(output, countLinks(nodes));
        for (int i = 0; i < nodes.size(); i++) {
            for (Node input : nodes.get(i).getInputs()) {
                if (input != null) {
                    kryo.writeObject(output, nodes.indexOf(input));
                    kryo.writeObject(output, i);
                }
            }
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
     * Should load the node's weights.
     * @param kryo the kryo configuration.
     * @param input the input.
     * @param node the node.
     * @param nodes the list.
     */
    private void loadWeightsRec(Kryo kryo, Input input, Node node, List<Node> nodes) {
        String tmp = kryo.readObject(input, String.class);
        node.loadWeights(kryo, input);
        Node[] children = node.getInputs();
        for (Node child : children) {
            if (child != null && !nodes.contains(child)) {
                nodes.add(child);
                loadWeightsRec(kryo, input, child, nodes);
            }
        }
    }

    /**
     * Load the weights of the configuration, i.e. do nothing except reading on the input.
     * @param kryo the kryo configuration.
     * @param input the input.
     * @return this.
     */
    public GraphConf loadWeights(Kryo kryo, Input input) {
        // Fetch outputs' ids.
        int nbOutputIdx = kryo.readObject(input, int.class);
        for (int i = 0; i < nbOutputIdx; i++) {
            kryo.readObject(input, Integer.class);
        }
        // Load all nodes' weights.
        kryo.readObject(input, int.class);
        List<Node> nodes = new LinkedList<>();
        for (Map.Entry<Integer, Node> entry: outputs.entrySet()) {
            nodes.add(entry.getValue());
            loadWeightsRec(kryo, input, entry.getValue(), nodes);
        }
        // Fetch all links.
        int nbLinks = kryo.readObject(input, int.class);
        for (int i = 0; i < nbLinks; i++) {
            kryo.readObject(input, int.class);
            kryo.readObject(input, int.class);
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
        // Load outputs' ids.
        int outputsSize = kryo.readObject(input, int.class);
        List<Integer> outputsId = new LinkedList<>();
        for (int i = 0; i < outputsSize; i++) {
            outputsId.add(kryo.readObject(input, Integer.class));
        }
        // Load all nodes.
        int nbNodes = kryo.readObject(input, int.class);
        List<Node> nodes = new LinkedList<>();
        for (int i = 0; i < nbNodes; i++) {
            nodes.add(NodesFactory.create(kryo, input));
        }
        // Create all links.
        int nbLinks = kryo.readObject(input, int.class);
        for (int i = 0; i < nbLinks; i++) {
            int iId = kryo.readObject(input, int.class);
            int oId = kryo.readObject(input, int.class);
            nodes.get(oId).addInputs(nodes.get(iId));
        }
        // Set the outputs.
        conf.setOutputs(outputsId.stream().map(nodes::get).toArray(Node[]::new));
        return conf;
    }

    /**
     * Print the graph configuration.
     */
    public void print() {
        System.out.println("Number of output: " + outputs.size() + ".");
    }
}
