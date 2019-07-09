package cnn.data;

import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ArrayPtrFactory {

    public static ArrayPtr empty() {
        return empty(true);
    }

    public static ArrayPtr empty(boolean gpu) {
        return new ArrayPtr(gpu);
    }

    public static ArrayPtr empty(long[] shape, int elementSize) {
        return new ArrayPtr(shape, elementSize);
    }

    public static ArrayPtr fromData(CUdeviceptr data, long[] shape) {
        return new ArrayPtr(data, shape);
    }

    public static ArrayPtr fromData(INDArray data) {
        return fromData(data, false);
    }

    public static ArrayPtr fromData(INDArray data, boolean gpu) {
        return new ArrayPtr(data, gpu);
    }

    public static ArrayPtr fromData(int[] data, boolean gpu) {
        return new ArrayPtr(data, gpu);
    }

    public static ArrayPtr fromData(long[] data, boolean gpu) {
        return new ArrayPtr(data, gpu);
    }
}
