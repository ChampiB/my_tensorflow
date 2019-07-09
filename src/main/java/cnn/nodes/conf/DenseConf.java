package cnn.nodes.conf;

import cnn.nodes.enumerations.ActivationType;

public class DenseConf {

    private int outputSize;
    private ActivationType af;

    public DenseConf() {
        this(10, ActivationType.SIGMOID);
    }

    public DenseConf(int outputSize) {
        this(outputSize, ActivationType.SIGMOID);
    }

    public DenseConf(int outputSize, ActivationType af) {
        this.outputSize = outputSize;
        this.af = af;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public DenseConf setOutputSize(int outputSize) {
        this.outputSize = outputSize;
        return this;
    }

    public ActivationType getAf() {
        return af;
    }

    public DenseConf setAf(ActivationType af) {
        this.af = af;
        return this;
    }
}
