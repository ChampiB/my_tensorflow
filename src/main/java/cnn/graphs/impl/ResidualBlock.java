package cnn.graphs.impl;

import cnn.graphs.Graph;
import cnn.nodes.Node;
import cnn.nodes.NodesFactory;
import cnn.nodes.conf.Conv2dConf;
import cnn.nodes.conf.PadConf;

import static cnn.nodes.enumerations.ActivationType.NONE;
import static cnn.nodes.enumerations.ActivationType.RELU;
import static cnn.nodes.enumerations.PaddingType.SAME;

/**
 * Residual block.
 */
public class ResidualBlock extends Graph {

    public enum ShortcutMethod {
        SAME_SIZE,
        ZERO_PADDING,
        CONVOLUTION
    }

    private int outputSize;
    private ShortcutMethod method;

    /**
     * Create a residual block.
     */
    public ResidualBlock(int[][] filters, int[][] strides, ShortcutMethod method) {
        this.method = method;
        this.outputSize = filters[0][0];

        Node func = null;
        for (int i = 0; i < filters.length; i++) {
            Conv2dConf conf = new Conv2dConf()
                    .setFilters(filters[i])
                    .setPadding(SAME)
                    .setStrides(strides[i])
                    .setAf((i + 1 != filters.length) ? RELU : NONE);
            func = NodesFactory.create("Conv2d", conf).setInputs(func);
        }

        Node jump = createJumpNode(method);
        Node add = NodesFactory.create("Add").setInputs(func, jump);
        Node activation = NodesFactory.create("Activation", RELU).setInputs(add);
        getConf().setOutputs(activation);
    }

    /**
     * Create the appropriate identity function.
     * @param method the method defining which node to create.
     * @return the identity node.
     */
    private Node createJumpNode(ShortcutMethod method) {
        switch (method) {
            case ZERO_PADDING:
                return NodesFactory.create("Pad2d", new PadConf(1, outputSize, 0));
            case CONVOLUTION:
                Conv2dConf conf = new Conv2dConf(new int[]{outputSize, 1, 1}, new int[]{1, 1}).setAf(NONE);
                return NodesFactory.create("Conv2d", conf);
            default:
                return NodesFactory.create("Identity");
        }
    }

    /**
     * Print the residual block.
     */
    @Override
    public void print() {
        System.out.println("Type: Residual block");
        System.out.println("       X");
        System.out.println("       |---------#");
        System.out.println("       |         |");
        System.out.println("  Conv2d(RELU)   |");
        switch (method) {
            case ZERO_PADDING:
                System.out.println("       |         | Pad2d(" + outputSize + ", 0)");
            case CONVOLUTION:
                System.out.println("       |         | Conv2d(" + outputSize + "x1x1)");
            default:
                System.out.println("       |         | Identity");
        }
        System.out.println(" Conv2d(NONE)  |");
        System.out.println("       |         |");
        System.out.println("   Add2d()<------#");
        System.out.println("       |");
        System.out.println("Activation(RELU)");
        System.out.println("       |");
        System.out.println("       Y");
    }
}
