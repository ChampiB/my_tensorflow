package cnn.nodes.impl.cpu;

import cnn.data.ArrayPtrFactory;
import cnn.nodes.Node;
import cnn.nodes.enumerations.ActivationType;
import cnn.data.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.google.common.collect.ImmutableMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static cnn.nodes.enumerations.ActivationType.*;

public class Activation extends Node {

    private ArrayPtr x;
    private ActivationType type;

    public Activation() {
        this(NONE);
    }

    public Activation(ActivationType type) {
        this.type = type;
    }

    public Activation(Object type) {
        this((ActivationType) type);
    }

    // Map of the activation functions.
    private static final Map<ActivationType, Function<INDArray, INDArray>> afs = ImmutableMap.of(
            NONE, x -> x,
            SOFTMAX, Transforms::softmax,
            RELU, Transforms::relu,
            SIGMOID, Transforms::sigmoid
    );

    // Map of the derivative of the activation functions.
    private static final Map<ActivationType, BiFunction<INDArray, INDArray, INDArray>> dafs = ImmutableMap.of(
            NONE, (INDArray x, INDArray g) -> g.mul(Nd4j.ones(x.shape())),
            SOFTMAX, Activation::softMaxDerivative,
            RELU, (INDArray x, INDArray g) -> {
                BooleanIndexing.replaceWhere(x, 1, Conditions.greaterThan(0));
                BooleanIndexing.replaceWhere(x, 0, Conditions.lessThanOrEqual(0));
                return g.mul(x);
            },
            SIGMOID, (INDArray x, INDArray g) -> g.mul(Transforms.sigmoidDerivative(x))
    );

    private static INDArray softMaxDerivative(INDArray x, INDArray g) {
        INDArray s = Transforms.softmax(x);
        int size = (int) s.shape()[1];
        for (int i = 0; i < x.shape()[0]; i++) {
            INDArray res = Nd4j.create(size, size);
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                    res.putScalar(j, k, s.getFloat(i, k) * ((j == k ? 1 : 0) - s.getFloat(i, j)) * g.getFloat(j, k));
            x.putRow(i, res.sum(1));
        }
        return x;
    }

    /**
     * Get the activation function of the layer.
     * @return the function.
     */
    private static Function<INDArray, INDArray> get(ActivationType af) {
        return afs.get(af);
    }

    /**
     * Get the derivative of the activation function of the layer.
     * @return the derivative.
     */
    private static BiFunction<INDArray, INDArray, INDArray> derivative(ActivationType af) {
        return dafs.get(af);
    }

    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... xs) {
        x = xs[0].dup();
        return ArrayPtrFactory.fromData(get(type).apply(x.toCPU()));
    }

    static ArrayPtr gradient;
    @Override
    public ArrayPtr[] update(double lr, ArrayPtr... gradient) {
        return new ArrayPtr[]{ArrayPtrFactory.fromData(derivative(type).apply(x.toCPU(), gradient[0].toCPU()))};
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
        System.out.println("Type: Activation(cpu)");
        System.out.println("Function: " + type);
    }
}
