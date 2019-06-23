package cnn;

import cnn.layers.Layer;
import cnn.useful.ArrayPtr;
import org.junit.jupiter.api.Assertions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Consumer;
import java.util.function.Function;

import static cnn.NeuralNetwork.SSE;

public class TestHelper {

    private static double computeNumericalGradient(
            Layer layer, Consumer<Double> update, Double xVal, ArrayPtr x, ArrayPtr t
    ) {
        double epsilon = 0.01;
        update.accept(xVal - epsilon);
        ArrayPtr ym = layer.activation(x.dup(), false).dup();
        update.accept(xVal + epsilon);
        ArrayPtr yp = layer.activation(x.dup(), false).dup();
        update.accept(xVal);
        return (SSE(t.dup(), yp) - SSE(t.dup(), ym)) / (2 * epsilon);
    }

    public static double numericalGradient_wb(cnn.layers.impl.cpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getBw().putScalar(pos, xVal), layer.getBw().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_wb(cnn.layers.impl.gpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getBw().putScalar(pos, xVal), layer.getBw().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_w(cnn.layers.impl.cpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_w(cnn.layers.impl.gpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_w(cnn.layers.impl.cpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_w(cnn.layers.impl.gpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_i(cnn.layers.impl.cpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_i(cnn.layers.impl.gpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_i(cnn.layers.impl.gpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), x, t
        );
    }

    public static double numericalGradient_i(cnn.layers.impl.cpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), x, t
        );
    }

    public static ArrayPtr ones(long[] shape) {
        return new ArrayPtr(Nd4j.ones(shape));
    }

    public static ArrayPtr arrange(long[] shape, double n) {
        INDArray result = Nd4j.create(shape).ravel();
        for (int i = 0; i < result.length(); i++)
            result.putScalar(i, i);
        return new ArrayPtr(result.mul(n).reshape(shape));
    }

    public static ArrayPtr arrange(long[] shape) {
        return arrange(shape, 1);
    }

    public static INDArray generate4d(long[] shape, Function<int[], Float> function) {
        INDArray result = Nd4j.create(shape);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        result.putScalar(i, j, k, l, function.apply(new int[]{i, j, k, l}));
                    }
                }
            }
        }
        return result;
    }

    public static INDArray generate2d(long[] shape, Function<int[], Float> function) {
        INDArray result = Nd4j.create(shape);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                result.putScalar(i, j, function.apply(new int[]{i, j}));
            }
        }
        return result;
    }

    public static INDArray generate1d(long x, Function<int[], Float> function) {
        INDArray result = Nd4j.create(x);
        for (int i = 0; i < x; i++) {
            result.putScalar(i, function.apply(new int[]{i}));
        }
        return result;
    }

    public static ArrayPtr create(float[] data) {
        return new ArrayPtr(Nd4j.create(data));
    }

    public static ArrayPtr create(float[][] data) {
        return new ArrayPtr(Nd4j.create(data));
    }

    public static ArrayPtr create(float[][][][] data) {
        return new ArrayPtr(Nd4j.create(data));
    }

    public static void assertEquals(INDArray expected, INDArray actual, double tolerance) { // tolerance in percentage of the expected value
        Assertions.assertEquals(expected.shape().length, actual.shape().length);
        for (int i = 0; i < expected.shape().length; i++) {
            Assertions.assertEquals(expected.shape()[i], actual.shape()[i]);
        }
        actual = actual.ravel();
        expected = expected.ravel();
        for (int i = 0; i < actual.length(); i++) {
            double margin = (expected.getDouble(i) / 100) * tolerance;
            Assertions.assertTrue(Math.abs(expected.getDouble(i) - actual.getDouble(i)) <= margin);
        }
    }

    public static void assertEquals(INDArray expected, ArrayPtr actual, double tolerance) { // tolerance in percentage of the expected value
        assertEquals(expected, actual.toCPU(), tolerance);
    }

    public static void assertEquals(INDArray expected, INDArray actual) {
        assertEquals(expected, actual, 0);
    }

    public static void assertEquals(INDArray expected, ArrayPtr actual) {
        assertEquals(expected, actual.toCPU());
    }

    public static void assertEquals(ArrayPtr expected, INDArray actual) {
        assertEquals(expected.toCPU(), actual);
    }

    public static void assertEquals(ArrayPtr expected, ArrayPtr actual) {
        assertEquals(expected.toCPU(), actual.toCPU());
    }

    public static ArrayPtr computeGradient_i(cnn.layers.impl.cpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setBw(bw);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_i(cnn.layers.impl.gpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setBw(bw);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_i(cnn.layers.impl.cpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_i(cnn.layers.impl.gpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_wb(cnn.layers.impl.gpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x.dup(), true);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getBw().sub(bw.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return new ArrayPtr(gradient);
    }

    public static ArrayPtr computeGradient_wb(cnn.layers.impl.cpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x.dup(), true);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getBw().sub(bw.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return new ArrayPtr(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.layers.impl.gpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x.dup(), true);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return new ArrayPtr(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.layers.impl.cpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x.dup(), true);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return new ArrayPtr(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.layers.impl.gpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        return new ArrayPtr(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.layers.impl.cpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        return new ArrayPtr(gradient);
    }
}
