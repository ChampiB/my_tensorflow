package cnn.nodes.conf;

public class Pooling2dConf {

    private int[] kernel;

    public Pooling2dConf() {
        this(new int[]{2, 2});
    }

    public Pooling2dConf(int[] kernel) {
        this.kernel = kernel;
    }

    public int[] getKernel() {
        return kernel;
    }

    public Pooling2dConf setKernel(int[] kernel) {
        this.kernel = kernel;
        return this;
    }
}
