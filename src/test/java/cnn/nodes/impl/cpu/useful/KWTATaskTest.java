package cnn.nodes.impl.cpu.useful;

import cnn.ops.cpu.useful.KWTATask;
import cnn.useful.cpu.ThreadPool;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.jupiter.api.Assertions.*;

class KWTATaskTest {

    @Test
    void call() {
        INDArray inputs = Nd4j.create(new double[][][][]{{{{1, 1}, {1, 1}, {1, 1}}, {{2, 2}, {2, 2}, {2, 2}}, {{3, 3}, {3, 3}, {3, 3}}}});
        INDArray results = Nd4j.zeros(inputs.shape());
        int index = 0;
        int k = 2;

        ThreadPoolExecutor tp = ThreadPool.getInstance();
        Future<Boolean> future = tp.submit(new KWTATask(inputs, results, index, k));
        try {
            future.get();
            INDArray zero = results.slice(index).slice(0).ravel();
            for (int i = 0; i < zero.shape()[0]; i++)
                assertEquals(0, zero.getDouble(i), "All elements of the first feature map must be equal to zero.");
            INDArray two = results.slice(index).slice(1).ravel();
            for (int i = 0; i < two.shape()[0]; i++)
                assertEquals(2, two.getDouble(i), "All elements of the second feature map must be equal to two.");
            INDArray three = results.slice(index).slice(2).ravel();
            for (int i = 0; i < three.shape()[0]; i++)
                assertEquals(3, three.getDouble(i), "All elements of the third feature map must be equal to three.");
        } catch (Exception e) {
            fail();
        }
    }
}