package cnn.helpers;

import cnn.data.ArrayPtrFactory;
import cnn.graphs.Graph;
import cnn.nodes.Node;
import cnn.graphs.impl.NeuralNetwork;
import cnn.data.ArrayPtr;
import cnn.nodes.impl.cpu.Dense;
import org.junit.jupiter.api.Assertions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Consumer;
import java.util.function.Function;

import static cnn.graphs.impl.NeuralNetwork.SSE;

public class TestHelper {

    private static double computeNumericalGradient(
            Node node, Consumer<Double> update, Double xVal, ArrayPtr t, ArrayPtr... x
    ) {
        double epsilon = 0.01;
        update.accept(xVal - epsilon);
        ArrayPtr ym = node.activation(false, x).dup();
        update.accept(xVal + epsilon);
        ArrayPtr yp = node.activation(false, x).dup();
        update.accept(xVal);
        return (SSE(t.dup(), yp) - SSE(t.dup(), ym)) / (2 * epsilon);
    }

    public static double numericalGradient_wb(cnn.nodes.impl.cpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getBw().putScalar(pos, xVal), layer.getBw().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_wb(cnn.nodes.impl.gpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getBw().putScalar(pos, xVal), layer.getBw().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_w(cnn.nodes.impl.cpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_w(Graph graph, cnn.nodes.impl.cpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                graph, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_w(cnn.nodes.impl.gpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_w(cnn.nodes.impl.cpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_w(cnn.nodes.impl.gpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> layer.getW().putScalar(pos, xVal), layer.getW().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(Graph graph, FakeDense dense, int[] pos, ArrayPtr x, ArrayPtr t) {
        double epsilon = 0.01;

        dense.setPos(pos);
        dense.setDelta(-epsilon);
        ArrayPtr ym = graph.activation(false, x).dup();
        dense.setDelta(epsilon);
        ArrayPtr yp = graph.activation(false, x).dup();
        return  (SSE(t.dup(), yp) - SSE(t.dup(), ym)) / (2 * epsilon);
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.Pad2d node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.Pad2d node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.KWTA2d node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.KWTA2d node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.Identity node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.Identity node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.AvgPooling2d node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.AvgPooling2d node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double[] numericalGradient_i(cnn.nodes.impl.cpu.Add node, int[] pos, ArrayPtr t, ArrayPtr... xs) {
        double[] result = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            ArrayPtr x = xs[i];
            result[i] = computeNumericalGradient(
                    node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, xs
            );
        }
        return result;
    }

    public static double[] numericalGradient_i(cnn.nodes.impl.gpu.Add node, int[] pos, ArrayPtr t, ArrayPtr... xs) {
        double[] result = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            ArrayPtr x = xs[i];
            result[i] = computeNumericalGradient(
                    node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, xs
            );
        }
        return result;
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.Activation node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.Activation node, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                node, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.Conv2d layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.gpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static double numericalGradient_i(cnn.nodes.impl.cpu.Dense layer, int[] pos, ArrayPtr x, ArrayPtr t) {
        return computeNumericalGradient(
                layer, (xVal) -> x.toCPU().putScalar(pos, xVal), x.toCPU().getDouble(pos), t, x
        );
    }

    public static ArrayPtr ones(long[] shape) {
        return ArrayPtrFactory.fromData(Nd4j.ones(shape));
    }

    public static ArrayPtr arrange(long[] shape, double n) {
        INDArray result = Nd4j.create(shape).ravel();
        for (int i = 0; i < result.length(); i++)
            result.putScalar(i, i);
        return ArrayPtrFactory.fromData(result.mul(n).reshape(shape));
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
        return ArrayPtrFactory.fromData(Nd4j.create(data));
    }

    public static ArrayPtr create(float[][] data) {
        return ArrayPtrFactory.fromData(Nd4j.create(data));
    }

    public static ArrayPtr create(float[][][][] data) {
        return ArrayPtrFactory.fromData(Nd4j.create(data));
    }

    public static void assertEquals(INDArray expected, INDArray actual, double tolerance, double minMargin) { // tolerance in percentage of the expected value
        Assertions.assertEquals(expected.shape().length, actual.shape().length);
        for (int i = 0; i < expected.shape().length; i++) {
            Assertions.assertEquals(expected.shape()[i], actual.shape()[i]);
        }
        actual = actual.ravel();
        expected = expected.ravel();
        for (int i = 0; i < actual.length(); i++) {
            double margin = (Math.abs(expected.getDouble(i)) / 100) * tolerance;
            if (margin < minMargin)
                margin = minMargin;
            Assertions.assertTrue(Math.abs(expected.getDouble(i) - actual.getDouble(i)) <= margin);
        }
    }

    public static void assertEquals(INDArray expected, ArrayPtr actual, double tolerance, double minMargin) { // tolerance in percentage of the expected value
        assertEquals(expected, actual.toCPU(), tolerance, minMargin);
    }

    public static void assertEquals(INDArray expected, INDArray actual, double tolerance) { // tolerance in percentage of the expected value
        assertEquals(expected, actual, tolerance, 0);
    }

    public static void assertEquals(INDArray expected, ArrayPtr actual, double tolerance) { // tolerance in percentage of the expected value
        assertEquals(expected, actual.toCPU(), tolerance);
    }

    public static void assertEquals(ArrayPtr expected, ArrayPtr actual, double tolerance) { // tolerance in percentage of the expected value
        assertEquals(expected.toCPU(), actual.toCPU(), tolerance);
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

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.Pad2d node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(Graph graph, cnn.nodes.impl.cpu.Dense[] layers, int i, ArrayPtr t, ArrayPtr... x) {
        INDArray[] w = new INDArray[layers.length];
        for (int j = 0; j < layers.length; j++) {
            w[j] = layers[j].getW().dup();
        }
        ArrayPtr y = graph.activation(true, x);
        graph.update(1, NeuralNetwork.SSEGradient(t.dup(), y));
        for (int j = 0; j < layers.length; j++) {
            layers[j].setW(ArrayPtrFactory.fromData(w[j]));
        }
        return layers[i].getInputsGradients();
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.Pad2d node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.KWTA2d node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.KWTA2d node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.Identity node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.Identity node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.AvgPooling2d node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.AvgPooling2d node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr[] computeGradient_i(cnn.nodes.impl.cpu.Add node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y));
    }

    public static ArrayPtr[] computeGradient_i(cnn.nodes.impl.gpu.Add node, ArrayPtr t, ArrayPtr... x) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y));
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.Activation node, ArrayPtr x, ArrayPtr t) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.Activation node, ArrayPtr x, ArrayPtr t) {
        ArrayPtr y = node.activation(true, x);
        return node.update(1, NeuralNetwork.SSEGradient(t.dup(), y))[0];
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setBw(bw);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setBw(bw);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.cpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_i(cnn.nodes.impl.gpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x);
        ArrayPtr gradient = layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        layer.setW(w);
        return gradient;
    }

    public static ArrayPtr computeGradient_wb(cnn.nodes.impl.gpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x.dup());
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getBw().sub(bw.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return ArrayPtrFactory.fromData(gradient);
    }

    public static ArrayPtr computeGradient_wb(cnn.nodes.impl.cpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x.dup());
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getBw().sub(bw.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return ArrayPtrFactory.fromData(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.nodes.impl.gpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x.dup());
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return ArrayPtrFactory.fromData(gradient);
    }

    public static ArrayPtr computeGradient_w(Graph graph, Dense[] layers, int i, ArrayPtr x, ArrayPtr t) {
        INDArray[] w = new INDArray[layers.length];
        for (int j = 0; j < layers.length; j++) {
            w[j] = layers[j].getW().dup();
        }
        ArrayPtr y = graph.activation(true, x.dup());
        graph.update(1, NeuralNetwork.SSEGradient(t.dup(), y));
        INDArray gradient = layers[i].getW().sub(w[i]).mul(-1);
        for (int j = 0; j < layers.length; j++) {
            layers[j].setW(ArrayPtrFactory.fromData(w[j]));
        }
        return ArrayPtrFactory.fromData(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.nodes.impl.cpu.Conv2d layer, ArrayPtr x, ArrayPtr w, ArrayPtr bw, ArrayPtr t) {
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x.dup());
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        layer.setBw(bw);
        return ArrayPtrFactory.fromData(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.nodes.impl.gpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        return ArrayPtrFactory.fromData(gradient);
    }

    public static ArrayPtr computeGradient_w(cnn.nodes.impl.cpu.Dense layer, ArrayPtr x, ArrayPtr w, ArrayPtr t) {
        layer.setW(w);
        ArrayPtr y = layer.activation(true, x);
        layer.update(NeuralNetwork.SSEGradient(t.dup(), y), 1);
        INDArray gradient = layer.getW().sub(w.toCPU()).mul(-1);
        layer.setW(w);
        return ArrayPtrFactory.fromData(gradient);
    }
}
