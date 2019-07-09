package cnn.nodes.impl.gpu;

import cnn.nodes.Node;
import cnn.nodes.enumerations.ActivationType;
import cnn.ops.OperationInterface;
import cnn.ops.OpsFactory;
import cnn.data.ArrayPtr;
import cnn.useful.gpu.GPUNode;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.google.common.collect.ImmutableMap;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static cnn.nodes.enumerations.ActivationType.*;

public class Activation extends GPUNode {

    private ArrayPtr x;
    private ActivationType type;
    private OperationInterface op = OpsFactory.create("Operation", "gpu");

    /**
     * Default constructor.
     */
    public Activation() {
        this(NONE);
    }

    /**
     * Constructor.
     * @param type the activation type, i.e. RELU, SIGMOID, ...
     */
    public Activation(ActivationType type) {
        super(GPUNode.NODES_PATH, "activation.cu", new String[]{"relu", "sigmoid", "softMax", "noneDerivative", "reluDerivative", "sigmoidDerivative", "softMaxDerivative"});
        this.type = type;
    }

    /**
     * Constructor.
     * @param type the activation type, i.e. RELU, SIGMOID, ...
     */
    public Activation(Object type) {
        this((ActivationType) type);
    }

    // Map of the activation functions.
    private final Map<ActivationType, Function<ArrayPtr, ArrayPtr>> afs = ImmutableMap.of(
            NONE, x -> x,
            RELU, x -> applyFunction("relu", x),
            SOFTMAX, x -> applyFunction("softMax", x, (int)x.getShape()[0], (int)x.getShape()[1]),
            SIGMOID, x -> applyFunction("sigmoid", x)
    );

    // Map of the derivative of the activation functions.
    private final Map<ActivationType, BiFunction<ArrayPtr, ArrayPtr, ArrayPtr>> dafs = ImmutableMap.of(
            NONE, (x, g) -> {
                applyFunction("noneDerivative", x);
                op.mul(x, g);
                return x;
            },
            RELU, (x, g) -> {
                applyFunction("reluDerivative", x);
                op.mul(x, g);
                return x;
            },
            SOFTMAX, (x, g) -> applyFunction(
                    "softMaxDerivative", x, g, (int)x.getShape()[0], (int)x.getShape()[1]
            ),
            SIGMOID, (x, g) -> {
                applyFunction("sigmoidDerivative", x);
                op.mul(x, g);
                return x;
            }
    );

    /**
     * Apply the function to the data.
     * @param name the function's name.
     * @param x the data.
     * @param g the gradients.
     * @return the output.
     */
    private ArrayPtr applyFunction(String name, ArrayPtr x, ArrayPtr g, int gridSize, int blockSize) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{x.getSize()}), x.toPTR(), g.toPTR());
        execute(name, parameters, gridSize, 1, 1, blockSize, 1, 1, blockSize * Sizeof.FLOAT);
        return x;
    }

    /**
     * Apply the function to the data.
     * @param name the function's name.
     * @param x the data.
     * @return the output.
     */
    private ArrayPtr applyFunction(String name, ArrayPtr x) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{x.getSize()}), x.toPTR());
        execute(name, parameters, x.getSize(), 1, 1, 1, 1, 1, 0);
        return x;
    }

    /**
     * Apply the function to the data.
     * @param name the function's name.
     * @param x the data.
     * @return the output.
     */
    private ArrayPtr applyFunction(String name, ArrayPtr x, int gridSize, int blockSize) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{x.getSize()}), x.toPTR());
        execute(name, parameters, gridSize, 1, 1, blockSize, 1, 1, blockSize * Sizeof.FLOAT);
        return x;
    }

    /**
     * Get the activation function of the layer.
     * @return the function.
     */
    private Function<ArrayPtr, ArrayPtr> get(ActivationType af) {
        return afs.get(af);
    }

    /**
     * Get the derivative of the activation function of the layer.
     * @return the derivative.
     */
    private BiFunction<ArrayPtr, ArrayPtr, ArrayPtr> derivative(ActivationType af) {
        return dafs.get(af);
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... xs) {
        x = xs[0].dup();
        return get(type).apply(xs[0].dup());
    }

    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        return new ArrayPtr[]{derivative(type).apply(x, gradient[0])};
    }

    @Override
    public void save(Kryo kryo, Output output) {
        kryo.writeObject(output, "Activation");
        kryo.writeObject(output, type.ordinal());
    }

    @Override
    public Node loadWeights(Kryo kryo, Input input) {
        kryo.readObject(input, int.class);
        return this;
    }

    @Override
    public Node load(Kryo kryo, Input input) {
        type = ActivationType.values()[kryo.readObject(input, int.class)];
        return this;
    }

    @Override
    public void print() {
        System.out.println("Type: Activation(gpu)");
        System.out.println("Function: " + type);
    }
}
