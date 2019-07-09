package cnn.helpers;

import cnn.data.ArrayPtrFactory;
import cnn.graphs.Graph;
import cnn.graphs.conf.GraphConf;
import cnn.nodes.Node;
import cnn.nodes.impl.cpu.Add;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static cnn.helpers.TestHelper.create;
import static cnn.nodes.enumerations.ActivationType.NONE;

public class TestGraphHelper {

    /**
     * Create a sequential graph with one input and one output.
     * @param inputSize the size of an input sample.
     * @return the graph and the list of dense layer.
     */
    public static Pair<Graph, FakeDense[]> createS11(int inputSize) {
        // Create nodes.
        FakeDense[] nodes = new FakeDense[] {
                new FakeDense(4, NONE),
                new FakeDense(3, NONE),
                new FakeDense(2, NONE)
        };

        // Create connectivity.
        nodes[1].setInputs(nodes[0]);
        nodes[2].setInputs(nodes[1]);

        // Set weights.
        nodes[0].setW(ArrayPtrFactory.fromData(Nd4j.ones(new long[]{inputSize + 1, 4})));
        nodes[1].setW(create(new float[][]{{0, 0, 0}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}}));
        nodes[2].setW(create(new float[][]{{0, 0}, {1, 2}, {1, 2}, {1, 2}}));

        // Create graph.
        GraphConf conf = new GraphConf().setOutputs(nodes[2]);
        return new Pair<>(new Graph(conf), nodes);
    }

    /**
     * Create a parallel graph with one input and one output.
     * @param inputSize The number of input samples.
     * @return the graph and the list of dense layer.
     */
    public static Pair<Graph, FakeDense[]> createP11(int inputSize) {
        // Create nodes.
        FakeDense[] nodes = new FakeDense[] {
                new FakeDense(3, NONE),
                new FakeDense(2, NONE),
                new FakeDense(2, NONE)
        };

        // Create connectivity.
        nodes[1].setInputs(nodes[0]);
        nodes[2].setInputs(nodes[0]);
        Node add = new Add().setInputs(nodes[1], nodes[2]);

        // Set weights.
        nodes[0].setW(ArrayPtrFactory.fromData(Nd4j.ones(new long[]{inputSize + 1, 3})));
        nodes[1].setW(create(new float[][]{{0, 0}, {1, 2}, {1, 2}, {1, 2}}));
        nodes[2].setW(create(new float[][]{{0, 0}, {1, 2}, {1, 2}, {1, 2}}));

        // Create graph.
        GraphConf conf = new GraphConf().setOutputs(add);
        return new Pair<>(new Graph(conf), nodes);
    }

    /**
     * Create a cyclic graph.
     * @return the graph and the list of dense layer.
     */
    public static Pair<Graph, FakeDense[]> createCyclic() {
        FakeDense[] nodes = new FakeDense[] {
                new FakeDense(4),
                new FakeDense(3),
                new FakeDense(2)
        };
        nodes[1].setInputs(nodes[0], nodes[2]);
        nodes[2].setInputs(nodes[1]);
        GraphConf conf = new GraphConf().setOutputs(nodes[2]);
        return new Pair<>(new Graph(conf), nodes);
    }

    /**
     * Create a non cyclic graph.
     * @return the graph and the list of dense layer.
     */
    public static Pair<Graph, FakeDense[]> createNonCyclic() {
        return createP11(10);
    }
}
