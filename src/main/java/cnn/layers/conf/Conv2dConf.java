package cnn.layers.conf;

import cnn.ops.cpu.Activation;

public class Conv2dConf implements LayerConf {

    private int[] filters;
    private int[] strides;
    private int k;
    private double ratio;
    private Activation.Type af;

    public Conv2dConf() {
        filters = new int[]{8, 2, 2};
        strides = new int[]{2, 2};
        af = Activation.Type.RELU;
        k = 0;
        ratio = 0;
    }

    public Conv2dConf(int k, double ratio) {
        this();
        this.k = k;
        this.ratio = ratio;
    }

    public int[] filters() {
        return filters;
    }

    public Conv2dConf setFilters(int[] filters) {
        this.filters = filters;
        return this;
    }

    public int[] strides() {
        return strides;
    }

    public Conv2dConf setStrides(int[] strides) {
        this.strides = strides;
        return this;
    }

    public Activation.Type activationFunction() {
        return af;
    }

    public Conv2dConf setActivationFunction(Activation.Type af) {
        this.af = af;
        return this;
    }

    public boolean useKWTA() {
        return k > 0;
    }

    public int k() {
        return k;
    }

    public Conv2dConf setK(int k) {
        this.k = k;
        return this;
    }

    public boolean useCPCA() {
        return ratio > 0;
    }

    public double ratio() {
        return ratio;
    }

    public Conv2dConf setRatio(double ratio) {
        this.ratio = ratio;
        return this;
    }
}
