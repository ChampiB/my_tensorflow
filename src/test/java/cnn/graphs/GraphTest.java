package cnn.graphs;

import cnn.data.ArrayPtr;
import cnn.helpers.FakeDense;
import cnn.nodes.impl.cpu.Dense;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import static cnn.helpers.TestGraphHelper.*;
import static cnn.helpers.TestHelper.*;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * This class is testing the Graph class.
 *
 * The test are named using the following format of abbreviations:
 *
 *     <Type_of_graph><Number_of_inputs><Number_of_outputs>
 *
 * The Type of graph is either S for sequential or P for parallel.
 *
 * For example a sequential graph with one input and two output will be denoted by: S12.
 *
 * The graph only contains dense (cpu) layer.
 *
 * Other layers have not been tested in the graph, because there are already tested individually for both CPU and GPU.
 */
class GraphTest {

    @Test
    void activationS11() {
        ArrayPtr x = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{72, 144}});

        Pair<Graph, FakeDense[]> pair = createS11(2);
        ArrayPtr y = pair.getFirst().activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void updateWeightsS11() {
        Pair<Graph, FakeDense[]> pair = createS11(2);

        for (int i = 0; i < pair.getSecond().length; i++) {
            ArrayPtr x = arrange(new long[]{3, 2});
            ArrayPtr t = arrange(new long[]{3, 2});

            Graph graph = pair.getFirst();
            Dense dense = pair.getSecond()[i];

            ArrayPtr gradient = computeGradient_w(graph, pair.getSecond(), i, x, t);
            INDArray tg = generate2d(dense.getW().shape(), (pos) -> (float)numericalGradient_w(graph, dense, pos, x, t));
            assertEquals(tg, gradient, 1);
        }
    }

    @Test
    void updateInputsS11() {
        Pair<Graph, FakeDense[]> pair = createS11(2);

        for (int i = 0; i < pair.getSecond().length; i++) {
            ArrayPtr x = arrange(new long[]{3, 2});
            ArrayPtr t = arrange(new long[]{3, 2});

            Graph graph = pair.getFirst();
            FakeDense dense = pair.getSecond()[i];

            ArrayPtr gradient = computeGradient_i(graph, pair.getSecond(), i, t, x);
            INDArray tg = generate2d(dense.getShapeOfX(), (pos) -> (float) numericalGradient_i(graph, dense, pos, x, t));
            assertEquals(tg, gradient, 1);
        }
    }

    @Test
    void activationP11() {
        ArrayPtr x = create(new float[][]{{1, 1}});
        ArrayPtr t = create(new float[][]{{18, 36}});

        Pair<Graph, FakeDense[]> pair = createP11(2);
        ArrayPtr y = pair.getFirst().activation(false, x);
        assertEquals(t, y);
    }

    @Test
    void updateWeightsP11() {
        Pair<Graph, FakeDense[]> pair = createP11(2);

        for (int i = 0; i < pair.getSecond().length; i++) {
            ArrayPtr x = arrange(new long[]{3, 2});
            ArrayPtr t = arrange(new long[]{3, 2});

            Dense dense = pair.getSecond()[i];

            ArrayPtr gradient = computeGradient_w(pair.getFirst(), pair.getSecond(), i, x, t);
            INDArray tg = generate2d(dense.getW().shape(), (pos) -> (float)numericalGradient_w(pair.getFirst(), dense, pos, x, t));
            assertEquals(tg, gradient, 1);
        }

    }

    @Test
    void updateInputsP11() {
        Pair<Graph, FakeDense[]> pair = createP11(2);

        for (int i = 0; i < pair.getSecond().length; i++) {
            ArrayPtr x = arrange(new long[]{3, 2});
            ArrayPtr t = arrange(new long[]{3, 2});

            Graph graph = pair.getFirst();
            FakeDense dense = pair.getSecond()[i];

            ArrayPtr gradient = computeGradient_i(graph, pair.getSecond(), i, t, x);
            INDArray tg = generate2d(dense.getShapeOfX(), (int[] pos) -> (float) numericalGradient_i(graph, dense, pos, x, t));
            assertEquals(tg, gradient, 1);
        }
    }

    @Test
    void cyclicMustFail() {
        try {
            createCyclic();
            fail();
        } catch (Exception e) {
            // The construction of a cyclic graph must fail.
        }
    }

    @Test
    void nonCyclicMustPass() {
        try {
            createNonCyclic(); // The construction of a non cyclic graph must pass.
        } catch (Exception e) {
            fail();
        }
    }
}
