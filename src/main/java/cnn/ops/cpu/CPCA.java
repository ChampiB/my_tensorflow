package cnn.ops.cpu;

import cnn.layers.conf.Conv2dConf;
import cnn.ops.CPCAInterface;
import cnn.useful.ArrayPtr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class CPCA implements CPCAInterface {

    private ArrayPtr dwcpca = new ArrayPtr(false);
    private ArrayPtr fc = new ArrayPtr(false);

    /**
     * Normalize the CPCA gradients.
     */
    private void computeMeanOfCPCA() {
        for (int fi = 0; fi < dwcpca.toCPU().shape()[0]; fi++) {
            double d = fc.toCPU().getNumber(fi).doubleValue();
            if (d != 0) {
                dwcpca.toCPU().putSlice(fi, dwcpca.toCPU().slice(fi).div(d));
            }
        }
    }

    /**
     * Compute the CPCA gradients.
     * @param x the input.
     * @param w the weights.
     * @param y the layer output.
     * @return the gradients.
     */
    public INDArray weightsGradients(Conv2dConf conf, ArrayPtr x, ArrayPtr w, ArrayPtr y) {
        dwcpca = new ArrayPtr(Nd4j.zeros(w.getShape()));
        fc = new ArrayPtr(Nd4j.zeros(y.getShape()[1]));
        for (int fi = 0; fi < y.getShape()[1]; fi++) {
            for (int ii = 0; ii < y.getShape()[0]; ii++) {
                INDArray weights = w.toCPU().slice(fi);
                int voff = 0;
                for (int vi = 0; vi < y.getShape()[2]; vi++) {
                    int hoff = 0;
                    for (int hi = 0; hi < y.getShape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[] {
                                NDArrayIndex.point(ii),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                                NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                        };
                        INDArray kernel = x.toCPU().get(indexes);
                        if (y.toCPU().getDouble(ii, fi, vi, hi) != 0) {
                            INDArray gwcpca = kernel.sub(weights).mul(Math.min(y.toCPU().getDouble(ii, fi, vi, hi), 1D));
                            dwcpca.toCPU().putSlice(fi, dwcpca.toCPU().slice(fi).add(gwcpca));
                            fc.toCPU().putScalar(fi, fc.toCPU().getNumber(fi).doubleValue() + 1);
                        }
                        hoff += conf.strides()[1];
                    }
                    voff += conf.strides()[0];
                }
            }
        }
        computeMeanOfCPCA();
        return dwcpca.toCPU();
    }
}
