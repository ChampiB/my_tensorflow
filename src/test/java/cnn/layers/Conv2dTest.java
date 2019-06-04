package cnn.layers;

import cnn.NeuralNetwork;
import cnn.layers.conf.ConfConv2d;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.sound.midi.Soundbank;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class Conv2dTest {

    @Test
    void activationWithBias() {
        INDArray w = Nd4j.create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        INDArray bw = Nd4j.create(new float[]{1});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{1, 2, 2})
                        .setStrides(new int[]{1, 1})
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, false);
        assertEquals(13, y.getDouble(0, 0, 0, 0));
        assertEquals(17, y.getDouble(0, 0, 0, 1));
        assertEquals(25, y.getDouble(0, 0, 1, 0));
        assertEquals(29, y.getDouble(0, 0, 1, 1));
    }

    @Test
    void activation() {
        INDArray w = Nd4j.create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        INDArray bw = Nd4j.create(new float[]{0});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{1, 2, 2})
                        .setStrides(new int[]{1, 1})
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, false);
        assertEquals(12, y.getDouble(0, 0, 0, 0));
        assertEquals(16, y.getDouble(0, 0, 0, 1));
        assertEquals(24, y.getDouble(0, 0, 1, 0));
        assertEquals(28, y.getDouble(0, 0, 1, 1));
    }

    @Test
    void activationManyFeatures() {
        INDArray w = Nd4j.create(new float[][][][]{{{{1, 1}, {1, 1}}}, {{{2, 2}, {2, 2}}}});
        INDArray bw = Nd4j.create(new float[]{0, 0});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{2, 2, 2})
                        .setStrides(new int[]{1, 1})
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, false);
        assertEquals(12, y.getDouble(0, 0, 0, 0));
        assertEquals(16, y.getDouble(0, 0, 0, 1));
        assertEquals(24, y.getDouble(0, 0, 1, 0));
        assertEquals(28, y.getDouble(0, 0, 1, 1));
        assertEquals(24, y.getDouble(0, 1, 0, 0));
        assertEquals(32, y.getDouble(0, 1, 0, 1));
        assertEquals(48, y.getDouble(0, 1, 1, 0));
        assertEquals(56, y.getDouble(0, 1, 1, 1));
    }

    @Test
    void activationKWTA() {
        INDArray w = Nd4j.create(new float[][][][]{{{{1, 1}, {1, 1}}},{{{2, 2}, {2, 2}}}});
        INDArray bw = Nd4j.create(new float[]{0, 0});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{2, 2, 2})
                        .setStrides(new int[]{1, 1})
                        .setK(1)
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, false);
        assertEquals(0, y.getDouble(0, 0, 0, 0));
        assertEquals(0, y.getDouble(0, 0, 0, 1));
        assertEquals(0, y.getDouble(0, 0, 1, 0));
        assertEquals(0, y.getDouble(0, 0, 1, 1));
        assertEquals(24, y.getDouble(0, 1, 0, 0));
        assertEquals(32, y.getDouble(0, 1, 0, 1));
        assertEquals(48, y.getDouble(0, 1, 1, 0));
        assertEquals(56, y.getDouble(0, 1, 1, 1));
    }

    @Test
    void activationWithBiasAndKWTA() {
        INDArray w = Nd4j.create(new float[][][][]{{{{1, 1}, {1, 1}}},{{{2, 2}, {2, 2}}}});
        INDArray bw = Nd4j.create(new float[]{1, 1});
        INDArray x = Nd4j.create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{2, 2, 2})
                        .setStrides(new int[]{1, 1})
                        .setK(1)
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, false);
        assertEquals(0, y.getDouble(0, 0, 0, 0));
        assertEquals(0, y.getDouble(0, 0, 0, 1));
        assertEquals(0, y.getDouble(0, 0, 1, 0));
        assertEquals(0, y.getDouble(0, 0, 1, 1));
        assertEquals(25, y.getDouble(0, 1, 0, 0));
        assertEquals(33, y.getDouble(0, 1, 0, 1));
        assertEquals(49, y.getDouble(0, 1, 1, 0));
        assertEquals(57, y.getDouble(0, 1, 1, 1));
    }

    double computeNumericalGradientBias(Conv2d layer, int[] wp, INDArray x, INDArray t) {
        double epsilon = 0.01;
        double w = layer.getBw().getDouble(wp);
        layer.getBw().putScalar(wp, w - epsilon);
        INDArray ym = layer.activation(x, false).dup();
        layer.getBw().putScalar(wp, w + epsilon);
        INDArray yp = layer.activation(x, false).dup();
        layer.getBw().putScalar(wp, w);
        return (NeuralNetwork.SSE(t, yp) - NeuralNetwork.SSE(t, ym)) / (2 * epsilon);
    }

    double computeNumericalGradient(Conv2d layer, int[] wp, INDArray x, INDArray t) {
        double epsilon = 0.01;
        double w = layer.getW().getDouble(wp);
        layer.getW().putScalar(wp, w - epsilon);
        INDArray ym = layer.activation(x, false).dup();
        layer.getW().putScalar(wp, w + epsilon);
        INDArray yp = layer.activation(x, false).dup();
        layer.getW().putScalar(wp, w);
        return (NeuralNetwork.SSE(t, yp) - NeuralNetwork.SSE(t, ym)) / (2 * epsilon);
    }

    double computeNumericalGradientInput(Conv2d layer, int[] wp, INDArray x, INDArray t) {
        double epsilon = 0.01;
        double xi = x.getDouble(wp);
        x.putScalar(wp, xi - epsilon);
        INDArray ym = layer.activation(x, false).dup();
        x.putScalar(wp, xi + epsilon);
        INDArray yp = layer.activation(x, false).dup();
        x.putScalar(wp, xi);
        return (NeuralNetwork.SSE(t, yp) - NeuralNetwork.SSE(t, ym)) / (2 * epsilon);
    }

    @Test
    void gradientReluInput() {
        INDArray w = Nd4j.create(new float[][][][]{
                {{{1, 1}, {1, 1}}},
                {{{2, 2}, {2, 2}}}
        });
        INDArray bw = Nd4j.create(new float[]{1, 1});
        INDArray x = Nd4j.create(new float[][][][]{{
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        }});
        INDArray t = Nd4j.create(new float[][][][]{{
                {{1, 1}, {1, 1}},
                {{1, 1}, {1, 1}}
        }});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{2, 2, 2})
                        .setStrides(new int[]{1, 1})
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, true);
        INDArray gradient = layer.update(NeuralNetwork.SSEGradient(t, y), 1).dup();
        layer.setBw(bw.dup());
        layer.setW(w.dup());

        for (int i = 0; i < x.shape()[1]; i++) {
            for (int j = 0; j < x.shape()[2]; j++) {
                for (int k = 0; k < x.shape()[3]; k++) {
                    double grad = computeNumericalGradientInput(layer, new int[]{0, i, j, k}, x, t);
                    assertTrue(
                            Math.abs(grad - gradient.getDouble(0, i, j, k)) < 0.1,
                            "Numerical gradient checking."
                    );
                }
            }
        }
    }

    @Test
    void gradientRelu() {
        INDArray w = Nd4j.create(new float[][][][]{
                {{{1, 1}, {1, 1}}},
                {{{2, 2}, {2, 2}}}
        });
        INDArray bw = Nd4j.create(new float[]{1, 1});
        INDArray x = Nd4j.create(new float[][][][]{{
                {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        }});
        INDArray t = Nd4j.create(new float[][][][]{{
                {{1, 1}, {1, 1}},
                {{1, 1}, {1, 1}}
        }});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{2, 2, 2})
                        .setStrides(new int[]{1, 1})
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        INDArray gradient = layer.getW().sub(w).mul(-1);
        layer.setW(w.dup());
        INDArray gradientBias = layer.getBw().sub(bw).mul(-1);
        layer.setBw(bw.dup());

        for (int i = 0; i < w.shape()[0]; i++) {
            for (int j = 0; j < w.shape()[1]; j++) {
                for (int k = 0; k < w.shape()[2]; k++) {
                    double grad = computeNumericalGradient(layer, new int[]{i, 0, j, k}, x, t);
                    assertTrue(
                            Math.abs(grad - gradient.getDouble(i, 0, j, k)) < 0.1,
                            "Numerical gradient checking."
                    );
                }
            }
            double grad = computeNumericalGradientBias(layer, new int[]{i}, x, t);
            assertTrue(
                    Math.abs(grad - gradientBias.getDouble(i)) < 0.1,
                    "Numerical gradient checking."
            );
        }
    }

    @Test
    void gradientReluAndKWTA() {
        INDArray w = Nd4j.create(new float[][][][]{
                {{{1, 1}, {1, 1}}},
                {{{2, 2}, {2, 2}}}
        });
        INDArray bw = Nd4j.create(new float[]{1, 1});
        INDArray x = Nd4j.create(new float[][][][]{{
            {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        }});
        INDArray t = Nd4j.create(new float[][][][]{{
            {{1, 1}, {1, 1}},
            {{1, 1}, {1, 1}}
        }});

        Conv2d layer = new Conv2d(
                new ConfConv2d()
                        .setFilters(new int[]{2, 2, 2})
                        .setStrides(new int[]{1, 1})
                        .setK(1)
        );
        layer.setBw(bw);
        layer.setW(w);

        INDArray y = layer.activation(x, true);
        layer.update(NeuralNetwork.SSEGradient(t, y), 1);
        INDArray gradient = layer.getW().sub(w).mul(-1);
        layer.setW(w.dup());
        INDArray gradientBias = layer.getBw().sub(bw).mul(-1);
        layer.setBw(bw.dup());

        for (int i = 0; i < w.shape()[0]; i++) {
            for (int j = 0; j < w.shape()[1]; j++) {
                for (int k = 0; k < w.shape()[2]; k++) {
                    double grad = computeNumericalGradient(layer, new int[]{i, 0, j, k}, x, t);
                    assertTrue(
                            Math.abs(grad - gradient.getDouble(i, 0, j, k)) < 0.1,
                            "Numerical gradient checking."
                    );
                }
            }
            double grad = computeNumericalGradientBias(layer, new int[]{i}, x, t);
            assertTrue(
                    Math.abs(grad - gradientBias.getDouble(i)) < 0.1,
                    "Numerical gradient checking."
            );
        }
    }
}
