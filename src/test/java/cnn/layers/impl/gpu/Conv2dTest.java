package cnn.layers.impl.gpu;

import cnn.useful.ArrayPtr;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static cnn.TestHelper.*;

class Conv2dTest {

    @Test
    void activationLargeImages() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}});
        ArrayPtr bw = create(new float[]{1});
        ArrayPtr x = arrange(new long[]{2, 2, 10, 10});
        ArrayPtr t = create(new float[][][][]{
         {{{  445f,  453f,  461f,  469f,  477f,  485f,  493f,  501f,  509f},
           {  525f,  533f,  541f,  549f,  557f,  565f,  573f,  581f,  589f},
           {  605f,  613f,  621f,  629f,  637f,  645f,  653f,  661f,  669f},
           {  685f,  693f,  701f,  709f,  717f,  725f,  733f,  741f,  749f},
           {  765f,  773f,  781f,  789f,  797f,  805f,  813f,  821f,  829f},
           {  845f,  853f,  861f,  869f,  877f,  885f,  893f,  901f,  909f},
           {  925f,  933f,  941f,  949f,  957f,  965f,  973f,  981f,  989f},
           { 1005f, 1013f, 1021f, 1029f, 1037f, 1045f, 1053f, 1061f, 1069f},
           { 1085f, 1093f, 1101f, 1109f, 1117f, 1125f, 1133f, 1141f, 1149f}}},
         {{{ 2045f, 2053f, 2061f, 2069f, 2077f, 2085f, 2093f, 2101f, 2109f},
           { 2125f, 2133f, 2141f, 2149f, 2157f, 2165f, 2173f, 2181f, 2189f},
           { 2205f, 2213f, 2221f, 2229f, 2237f, 2245f, 2253f, 2261f, 2269f},
           { 2285f, 2293f, 2301f, 2309f, 2317f, 2325f, 2333f, 2341f, 2349f},
           { 2365f, 2373f, 2381f, 2389f, 2397f, 2405f, 2413f, 2421f, 2429f},
           { 2445f, 2453f, 2461f, 2469f, 2477f, 2485f, 2493f, 2501f, 2509f},
           { 2525f, 2533f, 2541f, 2549f, 2557f, 2565f, 2573f, 2581f, 2589f},
           { 2605f, 2613f, 2621f, 2629f, 2637f, 2645f, 2653f, 2661f, 2669f},
           { 2685f, 2693f, 2701f, 2709f, 2717f, 2725f, 2733f, 2741f, 2749f}}}
        });

        Conv2d layer = new Conv2d(new int[]{1, 2, 2}, new int[]{1, 1});
        layer.setW(w);
        layer.setBw(bw);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void gradientReluLargeImages() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}, {{2, 2}, {2, 2}}}, {{{3, 3}, {3, 3}}, {{4, 4}, {4, 4}}}});
        ArrayPtr bw = create(new float[]{1, 1});
        ArrayPtr x = arrange(new long[]{2, 2, 10, 10}, 0.1);
        ArrayPtr t = ones(new long[]{2, 2, 9, 9});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1});
        ArrayPtr gradient = computeGradient_w(layer, x, w, bw, t);
        ArrayPtr gradientBias = computeGradient_wb(layer, x, w, bw, t);
        INDArray tg = generate4d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        INDArray tgb = generate1d(w.getShape()[0], (pos) -> (float)numericalGradient_wb(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
        assertEquals(tgb, gradientBias, 1);
    }

    @Test
    void activationWithBias() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        ArrayPtr bw = create(new float[]{1});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{13, 17}, {25, 29}}}});

        Conv2d layer = new Conv2d(new int[]{1, 2, 2}, new int[]{1, 1});
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void activation() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}}});
        ArrayPtr bw = create(new float[]{0});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{12, 16}, {24, 28}}}});

        Conv2d layer = new Conv2d(new int[]{1, 2, 2}, new int[]{1, 1});
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void activationManyFeatures() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}}, {{{2, 2}, {2, 2}}}});
        ArrayPtr bw = create(new float[]{0, 0});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{12, 16}, {24, 28}}, {{24, 32}, {48, 56}}}});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1});
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void activationKWTA() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}},{{{2, 2}, {2, 2}}}});
        ArrayPtr bw = create(new float[]{0, 0});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{0, 0}, {0, 0}}, {{24, 32}, {48, 56}}}});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1}, 1);
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void activationWithBiasAndKWTA() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}},{{{2, 2}, {2, 2}}}});
        ArrayPtr bw = create(new float[]{1, 1});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{0, 0}, {0, 0}}, {{25, 33}, {49, 57}}}});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1}, 1);
        layer.setBw(bw);
        layer.setW(w);
        ArrayPtr y = layer.activation(x, false);
        assertEquals(t, y);
    }

    @Test
    void gradientReluInput() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}}, {{{2, 2}, {2, 2}}}});
        ArrayPtr bw = create(new float[]{1, 1});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1});
        ArrayPtr gradient = computeGradient_i(layer, x, w, bw, t);
        INDArray tg = generate4d(x.getShape(), (pos) -> (float)numericalGradient_i(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
    }

    @Test
    void gradientRelu() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}}, {{{2, 2}, {2, 2}}}});
        ArrayPtr bw = create(new float[]{1, 1});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1});
        ArrayPtr gradient = computeGradient_w(layer, x, w, bw, t);
        ArrayPtr gradientBias = computeGradient_wb(layer, x, w, bw, t);
        INDArray tg = generate4d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        INDArray tgb = generate1d(w.getShape()[0], (pos) -> (float)numericalGradient_wb(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
        assertEquals(tgb, gradientBias, 1);
    }

    @Test
    void gradientReluAndKWTA() {
        ArrayPtr w = create(new float[][][][]{{{{1, 1}, {1, 1}}}, {{{2, 2}, {2, 2}}}});
        ArrayPtr bw = create(new float[]{1, 1});
        ArrayPtr x = create(new float[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        ArrayPtr t = create(new float[][][][]{{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}});

        Conv2d layer = new Conv2d(new int[]{2, 2, 2}, new int[]{1, 1}, 1);
        ArrayPtr gradient = computeGradient_w(layer, x, w, bw, t);
        ArrayPtr gradientBias = computeGradient_wb(layer, x, w, bw, t);
        INDArray tg = generate4d(w.getShape(), (pos) -> (float)numericalGradient_w(layer, pos, x, t));
        INDArray tgb = generate1d(w.getShape()[0], (pos) -> (float)numericalGradient_wb(layer, pos, x, t));
        assertEquals(tg, gradient, 1);
        assertEquals(tgb, gradientBias, 1);
    }
}
