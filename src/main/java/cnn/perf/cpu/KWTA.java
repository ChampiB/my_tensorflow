package cnn.perf.cpu;

import cnn.layers.conf.ConfConv2d;
import cnn.perf.KWTAInterface;
import cnn.perf.cpu.useful.KWTATask;
import cnn.perf.cpu.useful.ThreadPool;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Future;

public class KWTA implements KWTAInterface {

    /**
     * Compute the k-winners-take-all activation.
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output.
     */
    public INDArray activation(ConfConv2d conf, INDArray x) {
        // Launch one convolution task for each image.
        INDArray result = Nd4j.zeros(x.shape());
        List<Future<Boolean>> tasks = new LinkedList<>();
        for (int ii = 0; ii < x.shape()[0]; ii++) {
            tasks.add(ThreadPool.getInstance().submit(new KWTATask(x, result, ii, conf.k())));
        }
        ThreadPool.waitAll(tasks);
        return result;
    }
}
