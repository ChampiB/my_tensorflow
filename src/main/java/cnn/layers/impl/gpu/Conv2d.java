package cnn.layers.impl.gpu;

import cnn.layers.Layer;
import cnn.ops.*;
import cnn.layers.conf.Conv2dConf;
import cnn.useful.ArrayPtr;
import cnn.useful.gpu.GPUTask;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import jcuda.Pointer;
import jcuda.Sizeof;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class Conv2d extends GPUTask implements Layer {

    private ArrayPtr bwgpu = new ArrayPtr();
    private ArrayPtr wgpu = new ArrayPtr();
    private ArrayPtr cgpu = new ArrayPtr();
    private ArrayPtr yagpu = new ArrayPtr();
    private ArrayPtr yigpu = new ArrayPtr();
    private ArrayPtr ywgpu = new ArrayPtr();
    private ArrayPtr input = new ArrayPtr();

    private Conv2dConf conf;

    private KWTAInterface kwta = OpsFactory.create("KWTA", "gpu");
    private CPCAInterface cpca = OpsFactory.create("CPCA", "gpu");
    private OperationInterface op = OpsFactory.create("Operation", "gpu");
    private ActivationInterface layerActivation = OpsFactory.create("Activation", "gpu");

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
        super(GPUTask.LAYERS_PATH, "conv_2d.cu", new String[]{"activation", "inputs_gradients", "weights_gradients"});
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
     * Create randomly initialized weights.
     * @param xShape the input's shape.
     */
    private void createWeights(long[] xShape) {
        long[] shape = new long[]{conf.filters()[0], xShape[1], conf.filters()[1], conf.filters()[2]};
        wgpu.copy(Nd4j.rand(shape).mul(2).sub(1));
        shape = new long[]{conf.filters()[0]};
        bwgpu.copy(Nd4j.rand(shape).mul(2).sub(1));
    }

    /**
     * Compute the output shape.
     * @param conf the layer configuration.
     * @param xShape the input' shape.
     * @return the output shape.
     */
    private long[] computeOutputShape(Conv2dConf conf, long[] xShape) {
        // Compute the number of vertical and horizontal position.
        long nr = xShape[2] - conf.filters()[1] + 1;
        nr = (long) Math.ceil(((double)nr) / ((double)conf.strides()[0]));
        long nc = xShape[3] - conf.filters()[2] + 1;
        nc = (long) Math.ceil(((double)nc) / ((double)conf.strides()[1]));
        // Format the output.
        return new long[]{xShape[0], conf.filters()[0], nr, nc};
    }

    /**
     * Compute the output size of the layer.
     * @param conf the layer configuration.
     * @param xShape the input' shape.
     * @return the output size.
     */
    private int computeOutputSize(Conv2dConf conf, long[] xShape) {
        long[] shape = computeOutputShape(conf, xShape);
        return (int)(shape[0] * shape[1] * shape[2] * shape[3]);
    }

    /**
     * Compute the output size (gradients).
     * @param shape the output shape.
     * @return the output size.
     */
    private int computeOutputSize(long[] shape) {
        return (int)(shape[0] * shape[1] * shape[2] * shape[3]);
    }

    /**
     * Create the configuration.
     * @param conf the layer configuration.
     * @param wshape the weights shape.
     * @param xshape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(Conv2dConf conf, long[] wshape, long[] xshape, int nbElements, long[] yshape) {
        return new int[]{
                conf.filters()[0], conf.filters()[1], conf.filters()[2],
                conf.strides()[0], conf.strides()[1],
                (int)xshape[1], (int)xshape[2], (int)xshape[3],
                nbElements,
                (int)yshape[0], (int)yshape[1], (int)yshape[2], (int)yshape[3],
                (int)wshape[0], (int)wshape[1], (int)wshape[2], (int)wshape[3],
                (int)(yshape[1] * yshape[2] * yshape[3]),
                (int)(yshape[2] * yshape[3]),
                (int)(yshape[3]),
                (int)(xshape[1] * xshape[2] * xshape[3]),
                (int)(xshape[2] * xshape[3]),
                (int)(xshape[3])
        };
    }

    /**
     * Compute the convolution of the input with respect to the weights.
     * @param conf the layer configuration.
     * @param x the input.
     * @param w the weights.
     * @param bw the bias weights.
     */
    public void activationKernel(Conv2dConf conf, ArrayPtr x, ArrayPtr w, ArrayPtr bw) {
        // Allocate the output and configuration on device memory.
        long[] shape = computeOutputShape(conf, x.getShape());
        int size = computeOutputSize(conf, x.getShape());
        if (yagpu.isNull())
            yagpu = new ArrayPtr(shape, Sizeof.FLOAT);
        cgpu.copy(createConf(conf, w.getShape(), x.getShape(), size, shape));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), w.toPTR(), bw.toPTR(), yagpu.toPTR());
        execute(
                "activation", parameters,
                (int)shape[1], (int)shape[2], (int)shape[3],
                (int)shape[0], 1, 1,
                size
        );
    }

    /**
     * Compute the gradients with respect to the inputs.
     * @param conf the layer configuration.
     * @param yShape the output shape.
     * @param w the weights.
     * @param g the gradients with respect to the output.
     */
    public void inputsGradients(Conv2dConf conf, long[] yShape, ArrayPtr w, ArrayPtr g) {
        // Allocate the output and configuration on device memory.
        int size = computeOutputSize(yShape);
        if (yigpu.isNull())
            yigpu = new ArrayPtr(yShape, Sizeof.FLOAT);
        cgpu.copy(createConf(conf, w.getShape(), yShape, size, g.getShape()));
        // Create kernel parameters.
        Pointer kernelParameters = Pointer.to(cgpu.toPTR(), w.toPTR(), g.toPTR(), yigpu.toPTR());
        execute("inputs_gradients", kernelParameters, size, 256, 256 * Sizeof.FLOAT);
    }

    /**
     * Compute the gradient with respect to the weights.
     * @param conf the layer configuration.
     * @param dwShape the output shape.
     * @param x the inputs.
     * @param g the gradients with respect to the output.
     */
    public void weightsGradients(Conv2dConf conf, long[] dwShape, ArrayPtr x, ArrayPtr g) {
        // Allocate the output and configuration on device memory.
        int size = computeOutputSize(dwShape);
        int[] config = createConf(conf, dwShape, x.getShape(), size, g.getShape());
        if (ywgpu.isNull())
            ywgpu = new ArrayPtr(dwShape, Sizeof.FLOAT);
        cgpu.copy(config);
        // Create kernel parameters.
        Pointer parameters = Pointer.to(cgpu.toPTR(), x.toPTR(), g.toPTR(), ywgpu.toPTR());
        execute(
                "weights_gradients", parameters,
                config[13] * config[14], config[15], config[16],
                5, 10, 10,
                500 * Sizeof.FLOAT
        );
    }

    /**
     * Getter.
     * @return the bias weights.
     */
    public INDArray getBw() {
        return bwgpu.toCPU();
    }

    /**
     * Setter.
     * @param bw the bias weights.
     */
    public void setBw(ArrayPtr bw) {
        bwgpu.copy(bw);
    }

    /**
     * Getter.
     * @return the weights.
     */
    public INDArray getW() {
        return wgpu.toCPU();
    }

    /**
     * Setter.
     * @param w the weights.
     */
    public void setW(ArrayPtr w) {
        wgpu.copy(w);
    }

    @Override
    public ArrayPtr activation(ArrayPtr x, boolean training) {
        if (wgpu.isNull() || bwgpu.isNull())
            createWeights(x.getShape());
        activationKernel(conf, x, wgpu, bwgpu);
        if (conf.useKWTA())
            yagpu = kwta.activation(conf, yagpu);
        if (training)
            input = x;
        yagpu = layerActivation.apply(conf.activationFunction(), yagpu);
        return yagpu;
    }

    @Override
    public ArrayPtr update(ArrayPtr gradient, double lr) {
        // Compute the derivative of cost function with respect to the net input, i.e. Z = sum(wi*xi).
        op.mul(gradient, layerActivation.derivative(conf.activationFunction(), yagpu));
        // Compute the gradient, i.e. inputs, weights, bias weights and cpca.
        inputsGradients(conf, input.getShape(), wgpu, gradient);
        weightsGradients(conf, wgpu.getShape(), input, gradient);
        INDArray dbw = gradient.toCPU().sum(0, 2, 3);
        INDArray dwcpca;
        if (conf.useCPCA()) {
            dwcpca = cpca.weightsGradients(conf, input, wgpu, yagpu);
            op.mul(ywgpu, (float)(1 - conf.ratio()));
            op.add(ywgpu, new ArrayPtr(dwcpca.mul(conf.ratio())));
            dbw = dbw.mul(1 - conf.ratio());
        }
        // Update the weights.
        op.mul(ywgpu, (float)(-1 * lr));
        op.add(wgpu, ywgpu);
        op.add(bwgpu, new ArrayPtr(dbw.mul(-1 * lr), true));
        return yigpu;
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Conv2d");
        kryo.writeObject(output, conf.filters());
        kryo.writeObject(output, conf.strides());
        kryo.writeObject(output, conf.k());
        kryo.writeObject(output, conf.ratio());
        kryo.writeObject(output, conf.activationFunction());
        kryo.writeObject(output, wgpu.toCPU().data().asFloat());
        kryo.writeObject(output, wgpu.getShape());
        kryo.writeObject(output, bwgpu.toCPU().data().asFloat());
        kryo.writeObject(output, bwgpu.getShape());
    }

    @Override
    public Layer loadWeights(Kryo kryo, Input input) {
        conf.setFilters(kryo.readObject(input, int[].class));
        conf.setStrides(kryo.readObject(input, int[].class));
        kryo.readObject(input, int.class);
        kryo.readObject(input, double.class);
        kryo.readObject(input, ActivationInterface.Type.class);
        wgpu = new ArrayPtr(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        bwgpu = new ArrayPtr(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
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
        wgpu = new ArrayPtr(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        bwgpu = new ArrayPtr(Nd4j.create(kryo.readObject(input, float[].class)).reshape(kryo.readObject(input, long[].class)));
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Conv2d(gpu)");
        System.out.println("Filters: " + Arrays.toString(wgpu.getShape()));
        System.out.println("Strides: " + Arrays.toString(conf.strides()));
        System.out.println("Number of winners in kWTA: " + conf.k());
        System.out.println("Ratio (Back-propagation vs Hebbian): " + conf.ratio());
        System.out.println();
    }
}
