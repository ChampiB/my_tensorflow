package cnn.layers;

import cnn.layers.activation.Activation;
import cnn.layers.conf.ConfConv2d;
import cnn.perf.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Convolutional layer with CPCA and kWTA.
 */
public class Conv2d extends Layer {

    // Attributes.
    private INDArray w     = Nd4j.empty();
    private INDArray bw    = Nd4j.empty();
    private INDArray input = Nd4j.empty();
    private INDArray z     = Nd4j.empty();
    private INDArray y     = Nd4j.empty();
    private ConfConv2d conf;

    // Operators
    private Conv2dInterface conv2d;
    private KWTAInterface kwta;
    private CPCAInterface cpca;

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
        this.conv2d = TasksFactory.create("Conv2d");
        this.kwta = TasksFactory.create("KWTA");
        this.cpca = TasksFactory.create("CPCA");
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
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    public INDArray activation(INDArray x, boolean training) {
        if (w.isEmpty() || bw.isEmpty())
            createWeights(x.shape());
        z = conv2d.activation(conf, x, w, bw);
        if (conf.useKWTA())
            z = kwta.activation(conf, z);
        if (training)
            input = x;
        y = Activation.get(conf.activationFunction()).apply(z);
        return y;
    }

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    public INDArray update(INDArray gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi).
        gradient = gradient.mul(Activation.derivative(conf.activationFunction()).apply(z));
        // Compute the gradient, i.e. inputs, weights, bias weights and cpca.
        INDArray di = conv2d.inputsGradients(conf, input.shape(), w, gradient);
        INDArray dw = conv2d.weightsGradients(conf, w.shape(), input, gradient);
        INDArray dbw = gradient.sum(0, 2, 3);
        INDArray dwcpca;
        if (conf.useCPCA()) {
            dwcpca = cpca.weightsGradients(conf, input, w, y);
            dw = dw.mul(1 - conf.ratio()).add(dwcpca.mul(conf.ratio()));
            dbw = dbw.mul(1 - conf.ratio());
        }
        // Update the weights.
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
