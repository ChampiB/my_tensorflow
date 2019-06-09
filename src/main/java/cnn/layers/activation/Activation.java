package cnn.layers.activation;

import com.google.common.collect.ImmutableMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;
import java.util.function.Function;

import static cnn.layers.activation.Activation.Type.RELU;
import static cnn.layers.activation.Activation.Type.SIGMOID;

public class Activation {

    // Map of the activation functions.
    private static final Map<Type, Function<INDArray, INDArray>> afs = ImmutableMap.of(
            RELU, Transforms::relu,
            SIGMOID, Transforms::sigmoid
    );

    // Map of the derivative of the activation functions.
    private static final Map<Activation.Type, Function<INDArray, INDArray>> dafs = ImmutableMap.of(
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
    public static Function<INDArray, INDArray> get(Type af) {
        return afs.get(af);
    }

    /**
     * Get the derivative of the activation function of the layer.
     * @return the derivative.
     */
    public static Function<INDArray, INDArray> derivative(Type af) {
        return dafs.get(af);
    }

    public enum Type {
        SIGMOID,
        RELU
    }
}
