package cnn.ops.gpu;

import cnn.ops.ActivationInterface;
import cnn.useful.ArrayPtr;
import cnn.useful.gpu.GPUTask;
import com.google.common.collect.ImmutableMap;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Map;
import java.util.function.Function;

import static cnn.ops.ActivationInterface.Type.RELU;
import static cnn.ops.ActivationInterface.Type.SIGMOID;

public class Activation implements ActivationInterface {

    public Activation() {}

    // Map of the activation functions.
    private final Map<Type, Function<ArrayPtr, ArrayPtr>> afs = ImmutableMap.of(
            RELU, x -> applyFunction("relu", x),
            SIGMOID, x -> applyFunction("sigmoid", x)
    );

    // Map of the derivative of the activation functions.
    private final Map<Type, Function<ArrayPtr, ArrayPtr>> dafs = ImmutableMap.of(
            RELU, x -> applyFunction("reluDerivative", x),
            SIGMOID, x -> applyFunction("sigmoidDerivative", x)
    );

    private final Map<String, CUfunction> functions = GPUTask.loadFunctions(
            GPUTask.OPS_PATH, "activation.cu", new String[]{"relu", "sigmoid", "reluDerivative", "sigmoidDerivative"}
    );

    private ArrayPtr applyFunction(String name, ArrayPtr x) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{x.getSize()}), x.toPTR());
        GPUTask.execute(functions.get(name), parameters, x.getSize(), 1, 1, 1, 1, 1, 0);
        return x;
    }

    /**
     * Get the activation function of the layer.
     * @return the function.
     */
    private Function<ArrayPtr, ArrayPtr> get(Type af) {
        return afs.get(af);
    }

    /**
     * Get the derivative of the activation function of the layer.
     * @return the derivative.
     */
    private Function<ArrayPtr, ArrayPtr> derivative(Type af) {
        return dafs.get(af);
    }

    @Override
    public ArrayPtr apply(Type af, ArrayPtr x) {
        return get(af).apply(x);
    }

    @Override
    public ArrayPtr derivative(Type af, ArrayPtr x) {
        return derivative(af).apply(x);
    }
}
