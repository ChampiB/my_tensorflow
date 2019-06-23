package cnn.useful;

import cnn.useful.gpu.ContextHelper;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static jcuda.driver.JCudaDriver.*;

/**
 * This class hide the position of the data, i.e. cpu or gpu.
 */
public class ArrayPtr {

    private INDArray dataCPU = null;
    private CUdeviceptr dataGPU = null;
    private long[] shape = null;
    private boolean gpu;

    public ArrayPtr() {
        this(true);
    }

    public ArrayPtr(boolean gpu) {
        this.gpu = gpu;
        if (gpu)
            ContextHelper.initContext();
    }

    public ArrayPtr(long[] shape, int elementSize) {
        this(true);
        mallocGPU(shape, elementSize);
    }

    public ArrayPtr(INDArray data) {
        this(data, false);
    }

    public ArrayPtr(int[] data, boolean gpu) {
        this(gpu);
        copy(data);
    }

    public ArrayPtr(INDArray data, boolean gpu) {
        this(gpu);
        copy(data);
    }

    public boolean isNull() {
        return (gpu ? dataGPU : dataCPU) == null;
    }

    public void copy(int[] data) {
        shape = new long[]{data.length};
        if (gpu) {
            if (dataGPU == null) mallocGPU(shape, Sizeof.INT);
            cuMemcpyHtoD(dataGPU, Pointer.to(data), getSize() * Sizeof.INT);
        } else {
            dataCPU = Nd4j.create(toFloat(data));
        }
    }

    public void copy(INDArray data) {
        shape = data.shape();
        if (gpu) {
            if (dataGPU == null) mallocGPU(shape, Sizeof.FLOAT);
            cuMemcpyHtoD(dataGPU, Pointer.to(data.data().asFloat()), getSize() * Sizeof.FLOAT);
        } else {
            dataCPU = data;
        }
    }

    public void copy(ArrayPtr ptr) {
        shape = new long[ptr.getShape().length];
        System.arraycopy(ptr.getShape(), 0, shape, 0, ptr.getShape().length);
        if (gpu) {
            if (dataGPU == null) mallocGPU(shape, Sizeof.FLOAT);
            cuMemcpyDtoD(dataGPU, ptr.toGPU(), getSize() * Sizeof.FLOAT);
        } else {
            dataCPU = ptr.toCPU().dup();
        }
    }

    public CUdeviceptr toGPU() {
        if (!gpu) {
            ContextHelper.initContext();
            if (dataGPU == null) mallocGPU(shape, Sizeof.FLOAT);
            cuMemcpyHtoD(dataGPU, Pointer.to(dataCPU.data().asFloat()), getSize() * Sizeof.FLOAT);
            gpu = true;
        }
        return dataGPU;
    }

    public Pointer toPTR() {
        return Pointer.to(toGPU());
    }

    public INDArray toCPU() {
        if (gpu) {
            float[] buffer = new float[getSize()];
            cuMemcpyDtoH(Pointer.to(buffer), dataGPU, getSize() * Sizeof.FLOAT);
            dataCPU = Nd4j.create(buffer);
            freeGpu();
            gpu = false;
        }
        return (shape != null) ? dataCPU.reshape(shape) : dataCPU;
    }

    public void freeGpu() {
        if(dataGPU != null) cuMemFree(dataGPU);
        dataGPU = null;
    }

    public int getSize() {
        int size = 1;
        for (long e : shape)
            size *= e;
        return size;
    }

    public ArrayPtr dup() {
        ArrayPtr n = new ArrayPtr(gpu);
        n.copy(this);
        return n;
    }

    public long[] getShape() {
        return shape;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }

    private float[] toFloat(int[] data) {
        float[] newData = new float[data.length];
        for (int i = 0 ; i < data.length; i++)
            newData[i] = (float) data[i];
        return newData;
    }

    private void mallocGPU(long[] shape, int elementSize) {
        this.shape = shape;
        dataGPU = new CUdeviceptr();
        cuMemAlloc(dataGPU, getSize() * elementSize);
    }
}
