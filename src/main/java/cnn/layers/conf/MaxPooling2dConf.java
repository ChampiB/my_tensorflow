package cnn.layers.conf;

public class MaxPooling2dConf implements LayerConf {

    private int[] kernel;

    public MaxPooling2dConf() {
        this(new int[]{2, 2});
    }

    public MaxPooling2dConf(int[] kernel) {
        this.kernel = kernel;
    }

    public int[] getKernel() {
        return kernel;
    }

    public void setKernel(int[] kernel) {
        this.kernel = kernel;
    }
}
