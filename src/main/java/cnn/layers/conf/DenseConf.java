package cnn.layers.conf;

import cnn.ops.cpu.Activation;

public class DenseConf implements LayerConf {

    private int outputSize;
    private Activation.Type af;

    public DenseConf(int outputSize) {
        this(outputSize, Activation.Type.SIGMOID);
    }

    public DenseConf(int outputSize, Activation.Type af) {
        this.outputSize = outputSize;
        this.af = af;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public void setOutputSize(int outputSize) {
        this.outputSize = outputSize;
    }

    public Activation.Type getAf() {
        return af;
    }

    public void setAf(Activation.Type af) {
        this.af = af;
    }
}
