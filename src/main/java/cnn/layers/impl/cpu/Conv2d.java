package cnn.layers.impl.cpu;

import cnn.layers.Layer;
import cnn.ops.*;
import cnn.layers.conf.Conv2dConf;
import cnn.useful.ArrayPtr;
import cnn.layers.impl.cpu.useful.ConvolutionTask;
import cnn.useful.cpu.ThreadPool;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Future;

public class Conv2d implements Layer {

    // Attributes.
    private INDArray w     = Nd4j.empty();
    private INDArray bw    = Nd4j.empty();
    private INDArray input = Nd4j.empty();
    private INDArray z     = Nd4j.empty();
    private INDArray y     = Nd4j.empty();
    private Conv2dConf conf;

    // Operators
    private KWTAInterface kwta = OpsFactory.create("KWTA", "cpu");
    private CPCAInterface cpca = OpsFactory.create("CPCA", "cpu");
    private ActivationInterface layerActivation = OpsFactory.create("Activation", "cpu");

    /**
     * Default constructor.
     */
    public Conv2d() {
        this(new Conv2dConf());
    }

    /**
     * Constructor.
     */
    public Conv2d(int[] filters, int[] strides, int k) {
        this(new Conv2dConf().setFilters(filters).setStrides(strides).setK(k));
    }

    /**
     * Constructor.
     */
    public Conv2d(int[] filters, int[] strides) {
        this(new Conv2dConf().setFilters(filters).setStrides(strides));
    }

    /**
     * Constructor.
     * @param conf the layer's configuration.
     */
    public Conv2d(Conv2dConf conf) {
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
     * Constructor.
     * @param conf the layer's configuration.
     */
    public Conv2d(Object conf) {
        this((Conv2dConf) conf);
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param x the input.
     * @param w the weights.
     * @param bw the bias weights.
     * @return the output.
     */
    public INDArray activation(INDArray x, INDArray w, INDArray bw) {
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
        ThreadPool.waitAll(tasks);
        return result;
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray inputsGradients(long[] yShape, INDArray w, INDArray g) {
        INDArray di = Nd4j.zeros(yShape);
        for (int fi = 0; fi < g.shape()[1]; fi++) {
            for (int ii = 0; ii < g.shape()[0]; ii++) {
                INDArray weights = w.slice(fi);
                int voff = 0;
                for (int vi = 0; vi < g.shape()[2]; vi++) {
                    int hoff = 0;
                    for (int hi = 0; hi < g.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[] {
                                NDArrayIndex.point(ii),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                                NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                        };
                        INDArray gi = weights.mul(g.getDouble(ii, fi, vi, hi));
                        di.put(indexes, di.get(indexes).add(gi));
                        hoff += conf.strides()[1];
                    }
                    voff += conf.strides()[0];
                }
            }
        }
        return di;
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param yShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     * @return the gradients with respect to the inputs.
     */
    public INDArray weightsGradients(long[] yShape, INDArray x, INDArray g) {
        INDArray dw = Nd4j.zeros(yShape);
        for (int fi = 0; fi < g.shape()[1]; fi++) {
            for (int ii = 0; ii < g.shape()[0]; ii++) {
                int voff = 0;
                for (int vi = 0; vi < g.shape()[2]; vi++) {
                    int hoff = 0;
                    for (int hi = 0; hi < g.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[] {
                                NDArrayIndex.point(ii),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                                NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                        };
                        INDArray gw = x.get(indexes).mul(g.getDouble(ii, fi, vi, hi));
                        dw.putSlice(fi, dw.slice(fi).add(gw));
                        hoff += conf.strides()[1];
                    }
                    voff += conf.strides()[0];
                }
            }
        }
        return dw;
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
    public void setBw(ArrayPtr bw) {
        this.bw = bw.toCPU().dup();
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
    public void setW(ArrayPtr w) {
        this.w = w.toCPU().dup();
    }

    @Override
    public ArrayPtr activation(ArrayPtr x, boolean training) {
        if (w.isEmpty() || bw.isEmpty())
            createWeights(x.getShape());
        z = activation(x.toCPU(), w, bw);
        if (conf.useKWTA())
            z = kwta.activation(conf, new ArrayPtr(z)).toCPU();
        if (training)
            input = x.toCPU();
        y = layerActivation.apply(conf.activationFunction(), new ArrayPtr(z)).toCPU();
        return new ArrayPtr(y);
    }

    @Override
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi).
        INDArray grad = gradient.toCPU().mul(layerActivation.derivative(conf.activationFunction(), new ArrayPtr(z)).toCPU());
        // Compute the gradient, i.e. inputs, weights, bias weights and cpca.
        INDArray di = inputsGradients(input.shape(), w, grad);
        INDArray dw = weightsGradients(w.shape(), input, grad);
        INDArray dbw = grad.sum(0, 2, 3);
        INDArray dwcpca;
        if (conf.useCPCA()) {
            dwcpca = cpca.weightsGradients(conf, new ArrayPtr(input), new ArrayPtr(w), new ArrayPtr(y));
            dw = dw.mul(1 - conf.ratio()).add(dwcpca.mul(conf.ratio()));
            dbw = dbw.mul(1 - conf.ratio());
        }
        // Update the weights.
        w = w.add(dw.mul(-1 * lr));
        bw = bw.add(dbw.mul(-1 * lr));
        return new ArrayPtr(di);
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Conv2d");
        kryo.writeObject(output, conf.filters());
        kryo.writeObject(output, conf.strides());
        kryo.writeObject(output, conf.k());
        kryo.writeObject(output, conf.ratio());
        kryo.writeObject(output, conf.activationFunction());
        kryo.writeObject(output, w.data().asFloat());
        kryo.writeObject(output, w.shape());
        kryo.writeObject(output, bw.data().asFloat());
        kryo.writeObject(output, bw.shape());
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        conf.setFilters(kryo.readObject(input, int[].class));
        conf.setStrides(kryo.readObject(input, int[].class));
        kryo.readObject(input, int.class);
        kryo.readObject(input, double.class);
        kryo.readObject(input, ActivationInterface.Type.class);
        w = Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class));
        bw = Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class));
        return this;
    }

    @Override
    public Layer load(Kryo kryo, Input input) {
        conf = new Conv2dConf();
        conf.setFilters(kryo.readObject(input, int[].class));
        conf.setStrides(kryo.readObject(input, int[].class));
        conf.setK(kryo.readObject(input, int.class));
        conf.setRatio(kryo.readObject(input, double.class));
        conf.setActivationFunction(kryo.readObject(input, ActivationInterface.Type.class));
        w = Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class));
        bw = Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class));
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Conv2d(cpu)");
        System.out.println("Filters: " + Arrays.toString(w.shape()));
        System.out.println("Strides: " + Arrays.toString(conf.strides()));
        System.out.println("Number of winners in kWTA: " + conf.k());
        System.out.println("Ratio (Back-propagation vs Hebbian): " + conf.ratio());
        System.out.println();
    }
}
