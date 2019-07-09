package cnn.graphs.impl;

import cnn.nodes.Node;
import cnn.nodes.NodesFactory;
import cnn.nodes.conf.Conv2dConf;
import cnn.nodes.conf.Pooling2dConf;
import cnn.graphs.Graph;

import java.security.InvalidParameterException;

/**
 * Inception module.
 */
public class InceptionModule extends Graph {

    /**
     * Create a inception module.
     */
    public InceptionModule() {
        Node merge = NodesFactory.create("Merge2d").setInputs(
                createStack(
                        NodesFactory.create("Conv2d", new Conv2dConf().setFilters(new int[]{32, 1, 1}))
                ),
                createStack(
                        NodesFactory.create("Conv2d", new Conv2dConf().setFilters(new int[]{32, 1, 1})),
                        NodesFactory.create("Conv2d", new Conv2dConf().setFilters(new int[]{32, 3, 3}))
                ),
                createStack(
                        NodesFactory.create("Conv2d", new Conv2dConf().setFilters(new int[]{32, 1, 1})),
                        NodesFactory.create("Conv2d", new Conv2dConf().setFilters(new int[]{32, 5, 5}))
                ),
                createStack(
                        NodesFactory.create("MaxPooling2d", new Pooling2dConf().setKernel(new int[]{3, 3})),
                        NodesFactory.create("Conv2d", new Conv2dConf().setFilters(new int[]{32, 1, 1}))
                )
        );
        getConf().setOutputs(merge);
    }

    /**
     * Create a stack in the inception module.
     * @param nodes the nodes of the stack.
     * @return the last node of the stack.
     */
    public Node createStack(Node... nodes) {
        if (nodes.length == 0)
            throw new InvalidParameterException();
        for (int i = 0; i < nodes.length; i++) {
            if (i != 0)
                nodes[i].setInputs(nodes[i - 1]);
        }
        return nodes[nodes.length - 1];
    }

    /**
     * Print the inception module.
     */
    @Override
    public void print() {
        System.out.println("Type: Inception Module");
        System.out.println("       X");
        System.out.println("       |");
        System.out.println("       #-----------#-----------#-------------#");
        System.out.println("       |           |           |             |");
        System.out.println("       |      Conv2d(1x1) Conv2d(1x1) MaxPooling(3x3)");
        System.out.println("       |           |           |             |");
        System.out.println("  Conv2d(1x1) Conv2d(3x3) Conv2d(5x5)   Conv2d(1x1)");
        System.out.println("       |           |           |             |");
        System.out.println("       #-----------#-----------#-------------#");
        System.out.println("       |");
        System.out.println("    Merge2d");
        System.out.println("       |");
        System.out.println("       Y");
    }
}
