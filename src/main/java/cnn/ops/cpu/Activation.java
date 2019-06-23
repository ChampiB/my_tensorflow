package cnn.ops.cpu;

import cnn.ops.ActivationInterface;
import cnn.useful.ArrayPtr;
import com.google.common.collect.ImmutableMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;
import java.util.function.Function;

import static cnn.ops.ActivationInterface.Type.RELU;
import static cnn.ops.ActivationInterface.Type.SIGMOID;

public class Activation implements ActivationInterface {

    // Map of the activation functions.
    private static final Map<ActivationInterface.Type, Function<INDArray, INDArray>> afs = ImmutableMap.of(
            RELU, Transforms::relu,
            SIGMOID, Transforms::sigmoid
    );

    // Map of the derivative of the activation functions.
    private static final Map<ActivationInterface.Type, Function<INDArray, INDArray>> dafs = ImmutableMap.of(
            RELU, (INDArray x) -> {
                BooleanIndexing.replaceWhere(x, 1, Conditions.greaterThan(0));
                BooleanIndexing.replaceWhere(x, 0, Conditions.lessThanOrEqual(0));
                return x;
            },
            SIGMOID, Transforms::sigmoidDerivative
    );

    /**
     * Get the activation function of the layer.
     * @return the function.
     */
    private static Function<INDArray, INDArray> get(ActivationInterface.Type af) {
        return afs.get(af);
    }

    /**
     * Get the derivative of the activation function of the layer.
     * @return the derivative.
     */
    private static Function<INDArray, INDArray> derivative(ActivationInterface.Type af) {
        return dafs.get(af);
    }

    @Override
    public ArrayPtr apply(Type af, ArrayPtr x) {
        return new ArrayPtr(get(af).apply(x.toCPU()));
    }

    @Override
    public ArrayPtr derivative(Type af, ArrayPtr x) {
        return new ArrayPtr(derivative(af).apply(x.toCPU()));
    }
}
