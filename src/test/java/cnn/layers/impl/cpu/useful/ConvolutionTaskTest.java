package cnn.layers.impl.cpu.useful;

import cnn.TestHelper;
import cnn.layers.conf.Conv2dConf;
import cnn.useful.cpu.ThreadPool;
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
        Conv2dConf conf = new Conv2dConf().setFilters(new int[]{1, 2, 2}).setStrides(new int[]{1, 1});
        INDArray w = Nd4j.create(new double[][][]{{{1, 0}, {0, 1}}});
        INDArray bw = Nd4j.create(new double[]{0});
        INDArray t = Nd4j.create(new float[][][][]{{{{6, 8}, {12, 14}}}});

        ThreadPoolExecutor tp = ThreadPool.getInstance();
        Future<Boolean> future = tp.submit(new ConvolutionTask(input, output, 0, conf, w, bw));
        try {
            future.get();
            TestHelper.assertEquals(t, output);
        } catch (Exception e) {
            fail();
        }
    }
}