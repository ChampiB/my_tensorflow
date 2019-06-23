package cnn.ops.cpu;

import cnn.layers.conf.Conv2dConf;
import cnn.ops.KWTAInterface;
import cnn.ops.cpu.useful.KWTATask;
import cnn.useful.ArrayPtr;
import cnn.useful.cpu.ThreadPool;
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
    public ArrayPtr activation(Conv2dConf conf, ArrayPtr x) {
        // Launch one convolution task for each image.
        INDArray result = Nd4j.zeros(x.getShape());
        List<Future<Boolean>> tasks = new LinkedList<>();
        for (int ii = 0; ii < x.getShape()[0]; ii++) {
            tasks.add(ThreadPool.getInstance().submit(new KWTATask(x.toCPU(), result, ii, conf.k())));
        }
        ThreadPool.waitAll(tasks);
        return new ArrayPtr(result);
    }
}
