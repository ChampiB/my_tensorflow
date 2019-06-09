package cnn.layers.perf.cpu;

import cnn.layers.conf.ConfConv2d;
import cnn.layers.perf.CPCAInterface;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class CPCA implements CPCAInterface {

    private INDArray dwcpca;
    private INDArray fc;

    /**
     * Normalize the CPCA gradients.
     */
    private void computeMeanOfCPCA() {
        for (int fi = 0; fi < dwcpca.shape()[0]; fi++) {
            double d = fc.getNumber(fi).doubleValue();
            if (d != 0) {
                dwcpca.putSlice(fi, dwcpca.slice(fi).div(d));
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
    public INDArray weightsGradients(ConfConv2d conf, INDArray x, INDArray w, INDArray y) {
        dwcpca = Nd4j.zeros(w.shape());
        fc = Nd4j.zeros(y.shape()[1]);
        for (int fi = 0; fi < y.shape()[1]; fi++) {
            for (int ii = 0; ii < y.shape()[0]; ii++) {
                INDArray weights = w.slice(fi);
                int voff = 0;
                for (int vi = 0; vi < y.shape()[2]; vi++) {
                    int hoff = 0;
                    for (int hi = 0; hi < y.shape()[3]; hi++) {
                        INDArrayIndex[] indexes = new INDArrayIndex[] {
                                NDArrayIndex.point(ii),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(voff, voff + conf.filters()[2]),
                                NDArrayIndex.interval(hoff, hoff + conf.filters()[1])
                        };
                        INDArray kernel = x.get(indexes);
                        if (y.getDouble(ii, fi, vi, hi) != 0) {
                            INDArray gwcpca = kernel.sub(weights).mul(Math.min(y.getDouble(ii, fi, vi, hi), 1D));
                            dwcpca.putSlice(fi, dwcpca.slice(fi).add(gwcpca));
                            fc.putScalar(fi, fc.getNumber(fi).doubleValue() + 1);
                        }
                        hoff += conf.strides()[1];
                    }
                    voff += conf.strides()[0];
                }
            }
        }
        computeMeanOfCPCA();
        return dwcpca;
    }
}
