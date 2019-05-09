package cnn.layers;

import cnn.layers.activation.Activation;
import cnn.layers.conf.ConfConv2d;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import cnn.perf.ConvolutionTask;
import cnn.perf.ThreadPool;
import cnn.perf.KWTATask;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Future;

/**
 * Convolutional layer with CPCA and kWTA.
 */
public class Conv2d extends Layer {

    private INDArray w = null;
    private INDArray bw = null;
    private INDArray input = null;
    private INDArray z = null;
    private INDArray y = null;
    private ConfConv2d conf;

    private INDArray di = null;
    private INDArray dw = null;
    private INDArray dbw = null;
    private INDArray dwcpca = null;

    /**
     * Create a convolutional layer with CPCA and kWTA.
     * @param conf the layer configuration.
     */
    public Conv2d(ConfConv2d conf) {
        if (conf.filters().length != 3)
            throw new RuntimeException("Invalid filters's size: expected 3 got " + conf.filters().length);
        if (conf.strides().length != 2)
            throw new RuntimeException("Invalid strides's size: expected 2 got " + conf.strides().length);
        if (conf.useKWTA() && (conf.k() <= 0 || conf.k() > conf.filters()[0]))
            throw new RuntimeException("Invalid k: expected value between one and the number of filters got " + conf.k());
        if (conf.useCPCA() && (conf.ratio() < 0 || conf.ratio() > 1))
            throw new RuntimeException("Invalid ratio: expected value between zero and one got " + conf.ratio());
        this.conf = conf;
    }

    /**
     * Default constructor.
     */
    public Conv2d() {
        this(new ConfConv2d());
    }

    /**
     * Create randomly initialized weights.
     * @param xShape the input's shape.
     */
    private void createWeights(long[] xShape) {
        long[] shape = new long[]{conf.filters()[0], xShape[1], conf.filters()[1], conf.filters()[2]};
        w = Nd4j.rand(shape).mul(2).sub(1);
        shape = new long[]{conf.filters()[0]};
        bw = Nd4j.rand(shape).mul(2).sub(1);
    }

    /**
     * Wait for all tasks.
     * @param tasks the tasks to wait.
     */
    private void waitAll(List<Future<Boolean>> tasks) {
        for (Future<Boolean> task: tasks) {
            try {
                task.get();
            } catch (Exception e) {
                System.err.println("Unable to complete task: " + e.getMessage());
            }
        }
    }

    /**
     * This function apply the filters to the activation of the previous layer.
     * @param x the activation of the layer of size:
     *      [batches' size, number of channels, number of rows, number of columns].
     * @return the x's kernel of size:
     *      [batches' size, number of filters, number of horizontal position, number of vertical position].
     */
    private INDArray applyFilters(INDArray x) {
        // Compute the number of vertical and horizontal position.
        long nr = x.shape()[2] - conf.filters()[1] + 1;
        nr = (long) Math.ceil(((double)nr) / ((double)conf.strides()[0]));
        long nc = x.shape()[3] - conf.filters()[2] + 1;
        nc = (long) Math.ceil(((double)nc) / ((double)conf.strides()[1]));
        // Launch one convolution task for each image.
        INDArray result = Nd4j.zeros(x.shape()[0], conf.filters()[0], nr, nc);
        List<Future<Boolean>> tasks = new LinkedList<>();
        for (int ii = 0; ii < x.shape()[0]; ii++) {
            tasks.add(ThreadPool.getInstance().submit(new ConvolutionTask(x, result, ii, conf, w, bw)));
        }
        waitAll(tasks);
        return result;
    }

    /**
     * Apply the k-winners-take-all, i.e. neuronal competition.
     * @param y the layer's activation (before kWTA).
     * @return the kWTA activation.
     */
    private INDArray applyKWTA(INDArray y) {
        // Launch one convolution task for each image.
        INDArray result = Nd4j.zeros(y.shape());
        List<Future<Boolean>> tasks = new LinkedList<>();
        for (int ii = 0; ii < y.shape()[0]; ii++) {
            tasks.add(ThreadPool.getInstance().submit(new KWTATask(y, result, ii, conf.k())));
        }
        waitAll(tasks);
        return result;
    }

    /**
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    public INDArray activation(INDArray x, boolean training) {
        if (w == null || bw == null)
            createWeights(x.shape());
        z = applyFilters(x);
        if (conf.useKWTA())
            z = applyKWTA(z);
        if (training)
            input = x;
        y = Activation.get(conf.activationFunction()).apply(z);
        return y;
    }

    /**
     * Getter.
     * @param from the source.
     * @param ii the index of the image.
     * @param vi the index of the vertical position.
     * @param hi the index of the horizontal position.
     * @return the inputs of the neuron corresponding to the coordinates (ii, fi, vi, hi).
     */
    private INDArray getKernel(INDArray from, int ii, int vi, int hi) {
        return from.get(
                NDArrayIndex.indices(ii),
                NDArrayIndex.all(),
                NDArrayIndex.interval(vi, vi + conf.filters()[2]),
                NDArrayIndex.interval(hi, hi + conf.filters()[1])
        );
    }

    /**
     * Compute all the gradients (inputs, weights and CPCA).
     * @param gradient the gradient with
     * @param ii the image index.
     * @param fi the feature index.
     */
    private void computeGradients(INDArray gradient, int ii, int fi) {
        INDArray weights = w.slice(fi);
        int voff = 0;
        for (int vi = 0; vi < gradient.shape()[2]; vi++) {
            int hoff = 0;
            for (int hi = 0; hi < gradient.shape()[3]; hi++) {
                INDArrayIndex[] indexes = new INDArrayIndex[] {
                        NDArrayIndex.point(ii),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                        NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                };
                INDArray kernel = getKernel(input, ii, voff, hoff);
                INDArray gw = kernel.mul(gradient.getDouble(ii, fi, vi, hi));
                INDArray gi = weights.mul(gradient.getDouble(ii, fi, vi, hi));
                if (conf.useCPCA() && y.getDouble(ii, fi, vi, hi) != 0) {
                    INDArray gwcpca = kernel.sub(weights).mul(y.getDouble(ii, fi, vi, hi));
                    gw = gw.mul(1 - conf.ratio()).add(gwcpca.mul(conf.ratio()));
                }
                dw.putSlice(fi, dw.slice(fi).add(gw));
                di.put(indexes, getKernel(di, ii, voff, hoff).add(gi));
                hoff += conf.strides()[1];
            }
            voff += conf.strides()[0];
        }
    }

    /**
     * Compute all the gradients (inputs, weights and CPCA).
     * @param gradient the gradient with
     */
    private void computeGradients(INDArray gradient) {
        di = Nd4j.zeros(input.shape());
        dwcpca = (conf.useCPCA()) ? Nd4j.zeros(w.shape()) : null;
        dw = Nd4j.zeros(w.shape());
        for (int fi = 0; fi < gradient.shape()[1]; fi++) {
            for (int ii = 0; ii < gradient.shape()[0]; ii++) {
                computeGradients(gradient, ii, fi);
            }
        }
        dbw = gradient.sum(0, 2, 3);
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    public INDArray update(INDArray gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi)
        gradient = gradient.mul(Activation.derivative(conf.activationFunction()).apply(z));
        // Compute the derivative of cost function with respect to the inputs and the weights.
        computeGradients(gradient);
        // Update the weights
        if (conf.useCPCA()) {
            dw = dw.mul(1 - conf.ratio()).add(dwcpca.mul(conf.ratio()));
            dbw = dbw.mul(1 - conf.ratio());
        }
        w = w.add(dw.mul(-1 * lr));
        bw = bw.add(dbw.mul(-1 * lr));
        return di;
    }

    /**
     * Getter.
     * @return the bias weights.
     */
    public INDArray getBw() {
        return bw;
    }

    /**
     * Setter.
     * @param bw the bias weights.
     */
    public void setBw(INDArray bw) {
        this.bw = bw;
    }

    /**
     * Getter.
     * @return the weights.
     */
    public INDArray getW() {
        return w;
    }

    /**
     * Setter.
     * @param w the weights.
     */
    public void setW(INDArray w) {
        this.w = w;
    }

    /**
     * Display the layer on the standard output.
     */
    public void print() {
        System.out.println("Type: Conv2d");
        System.out.println("Filters: " + Arrays.toString(w.shape()));
        System.out.println("Strides: " + Arrays.toString(conf.strides()));
        System.out.println("Number of winners in kWTA: " + conf.k());
        System.out.println("Ratio (Back-propagation vs Hebbian): " + conf.ratio());
        System.out.println();
    }
}
