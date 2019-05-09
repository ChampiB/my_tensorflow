package cnn.layers.conf;

import cnn.layers.activation.Activation;

public class ConfConv2d {

    private int[] filters;
    private int[] strides;
    private int k;
    private double ratio;
    private Activation.Type af;

    public ConfConv2d() {
        filters = new int[]{8, 2, 2};
        strides = new int[]{2, 2};
        af = Activation.Type.RELU;
        k = 0;
        ratio = 0;
    }

    public ConfConv2d(int k, double ratio) {
        this();
        this.k = k;
        this.ratio = ratio;
    }

    public int[] filters() {
        return filters;
    }

    public ConfConv2d setFilters(int[] filters) {
        this.filters = filters;
        return this;
    }

    public int[] strides() {
        return strides;
    }

    public ConfConv2d setStrides(int[] strides) {
        this.strides = strides;
        return this;
    }

    public Activation.Type activationFunction() {
        return af;
    }

    public ConfConv2d setActivationFunction(Activation.Type af) {
        this.af = af;
        return this;
    }

    public boolean useKWTA() {
        return k > 0;
    }

    public int k() {
        return k;
    }

    public ConfConv2d setK(int k) {
        this.k = k;
        return this;
    }

    public boolean useCPCA() {
        return ratio > 0;
    }

    public double ratio() {
        return ratio;
    }

    public ConfConv2d setRatio(double ratio) {
        this.ratio = ratio;
        return this;
    }
}
