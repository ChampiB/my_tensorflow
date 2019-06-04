package cnn.perf.cpu.useful;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.concurrent.Callable;

public class KWTATask implements Callable<Boolean> {

    private INDArray input;
    private INDArray output;
    private int index;
    private int k;

    /**
     * Constructor.
     * @param input the input buffer (i.e. activation before kWTA).
     * @param output the output buffer (i.e. activation after kWTA).
     * @param index the example's index that must be processed.
     * @param k the number of winners.
     */
    public KWTATask(INDArray input, INDArray output, int index, int k) {
        this.input = input;
        this.output = output;
        this.index = index;
        this.k = k;
    }

    /**
     * Keep only the k largest (absolute) activation and set the other to zero.
     * @return true if the task success and false otherwise.
     */
    @Override
    public Boolean call() {
        for (int vi = 0; vi < output.shape()[2]; vi++) {
            for (int hi = 0; hi < output.shape()[3]; hi++) {
                INDArrayIndex[] indexes = new INDArrayIndex[] {
                        NDArrayIndex.point(index), NDArrayIndex.all(), NDArrayIndex.point(vi), NDArrayIndex.point(hi)
                };
                INDArray kernel = input.get(indexes).ravel();
                INDArray kernelKWTA = kernel.dup();
                for (int i = 0; i < kernel.shape()[0] - k; i++) {
                    int idx = Nd4j.getExecutioner().execAndReturn(new IAMin(kernelKWTA)).getFinalResult().intValue();
                    kernelKWTA.putScalar(idx, Double.MAX_VALUE);
                    kernel.putScalar(idx, 0);
                }
                output.put(indexes, kernel);
            }
        }
        return true;
    }
}