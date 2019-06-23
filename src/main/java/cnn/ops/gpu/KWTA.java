package cnn.ops.gpu;

import cnn.layers.conf.Conv2dConf;
import cnn.useful.ArrayPtr;
import cnn.ops.KWTAInterface;
import cnn.useful.gpu.GPUTask;
import jcuda.Pointer;
import jcuda.Sizeof;

public class KWTA extends GPUTask implements KWTAInterface {

    private ArrayPtr y = new ArrayPtr();
    private ArrayPtr c = new ArrayPtr();

    /**
     * Default constructor.
     */
    public KWTA() {
        super(GPUTask.OPS_PATH, "kwta.cu", new String[]{"activation"});
    }

    /**
     * Create the configuration.
     * @param shape the input shape.
     * @param nbElements the number of elements in the output array.
     * @return the configuration.
     */
    private int[] createConf(long[] shape, int nbElements) {
        return new int[]{
                (int)(shape[1] * shape[2] * shape[3]), (int)(shape[2] * shape[3]), (int)shape[3], nbElements
        };
    }

    /**
     * Compute the k-winners-take-all activation.
     * @param conf the layer configuration.
     * @param x the input.
     * @return the output.
     */
    public ArrayPtr activation(Conv2dConf conf, ArrayPtr x) {
        // Allocate the output and configuration on device memory.
        if (y.isNull())
            y = new ArrayPtr(x.getShape(), Sizeof.FLOAT);
        y.copy(x);
        c.copy(createConf(x.getShape(), x.getSize()));
        // Create kernel parameters.
        Pointer parameters = Pointer.to(c.toPTR(), Pointer.to(new int[]{conf.k()}), y.toPTR(), x.toPTR());
        execute(
                "activation", parameters,
                (int)x.getShape()[2], (int)x.getShape()[3], 1,
                (int)x.getShape()[0], 1, 1,
                0
        );
        return x;
    }
}
