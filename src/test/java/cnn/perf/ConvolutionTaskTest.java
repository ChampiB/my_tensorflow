package cnn.perf;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.cpu.useful.ConvolutionTask;
import cnn.perf.cpu.useful.ThreadPool;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.jupiter.api.Assertions.*;

class ConvolutionTaskTest {

    @Test
    void call() {
        INDArray input = Nd4j.create(new double[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
        INDArray output = Nd4j.zeros(1, 1, 2, 2);
        int index = 0;
        ConfConv2d conf = new ConfConv2d()
                .setFilters(new int[]{1, 2, 2})
                .setStrides(new int[]{1, 1});
        INDArray w = Nd4j.create(new double[][][]{{{1, 0}, {0, 1}}});
        INDArray bw = Nd4j.create(new double[]{0});

        ThreadPoolExecutor tp = ThreadPool.getInstance();
        Future<Boolean> future = tp.submit(new ConvolutionTask(input, output, index, conf, w, bw));
        try {
            future.get();
            assertEquals(6, output.getDouble(0, 0, 0, 0), "output[0][0] must be equal to 6.");
            assertEquals(8, output.getDouble(0, 0, 0, 1), "output[0][1] must be equal to 8.");
            assertEquals(12, output.getDouble(0, 0, 1, 0), "output[1][0] must be equal to 12.");
            assertEquals(14, output.getDouble(0, 0, 1, 1), "output[1][1] must be equal to 14.");
        } catch (Exception e) {
            fail();
        }
    }
}