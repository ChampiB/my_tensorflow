package cnn.nodes.conf;

import cnn.nodes.enumerations.ActivationType;
import cnn.nodes.enumerations.PaddingType;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

public class Conv2dConf {

    private int[] filters;
    private int[] strides;
    private int k;
    private double ratio;
    private ActivationType af;
    private PaddingType padding;

    public Conv2dConf() {
        filters = new int[]{8, 2, 2};
        strides = new int[]{2, 2};
        af = ActivationType.RELU;
        k = 0;
        ratio = 0;
        padding = PaddingType.VALID;
    }

    public Conv2dConf(int k, double ratio) {
        this();
        this.k = k;
        this.ratio = ratio;
    }

    public Conv2dConf(int[] filters, int[] strides) {
        this();
        this.filters = filters;
        this.strides = strides;
    }

    public int[] filters() {
        return filters;
    }

    public Conv2dConf setFilters(int[] filters) {
        this.filters = filters;
        return this;
    }

    public PaddingType padding() {
        return padding;
    }

    public Conv2dConf setPadding(PaddingType padding) {
        this.padding = padding;
        return this;
    }

    public int[] strides() {
        return strides;
    }

    public Conv2dConf setStrides(int[] strides) {
        this.strides = strides;
        return this;
    }

    public ActivationType getAf() {
        return af;
    }

    public Conv2dConf setAf(ActivationType af) {
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

    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, filters);
        kryo.writeObject(output, strides);
        kryo.writeObject(output, k);
        kryo.writeObject(output, ratio);
        kryo.writeObject(output, af);
        kryo.writeObject(output, padding);
    }

    public void loadWeights(Kryo kryo, Input input) {
        filters = kryo.readObject(input, int[].class);
        filters = kryo.readObject(input, int[].class);
        kryo.readObject(input, int.class);
        kryo.readObject(input, double.class);
        kryo.readObject(input, ActivationType.class);
        kryo.readObject(input, PaddingType.class);
    }

    public void load(Kryo kryo, Input input) {
        filters = kryo.readObject(input, int[].class);
        strides = kryo.readObject(input, int[].class);
        k = kryo.readObject(input, int.class);
        ratio = kryo.readObject(input, double.class);
        af = kryo.readObject(input, ActivationType.class);
        padding = kryo.readObject(input, PaddingType.class);
    }
}
