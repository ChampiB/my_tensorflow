package cnn.ops.gpu;

import cnn.data.ArrayPtrFactory;
import cnn.ops.OperationInterface;
import cnn.data.ArrayPtr;
import cnn.useful.gpu.GPUNode;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import java.util.Map;

public class Operation implements OperationInterface {

    private final Map<String, CUfunction> functions = GPUNode.loadFunctions(GPUNode.OPS_PATH, "operation.cu", new String[]{"mul_array", "mul_float", "add_array", "sub_array", "pow_array", "sum", "arg_max_row"});

    public Operation() {}

    @Override
    public int argMax(ArrayPtr a, int row) {
        ArrayPtr r = ArrayPtrFactory.empty(new long[]{1}, Sizeof.FLOAT);
        Pointer parameters = Pointer.to(Pointer.to(new int[]{(int) a.getShape()[1]}), Pointer.to(new int[]{row}), a.toPTR(), r.toPTR());
        GPUNode.execute(functions.get("arg_max_row"), parameters, 1, 1, 1, 1, 1, 1, 0);
        return r.toCPU().getInt(0);
    }

    @Override
    public float sum(ArrayPtr a) {
        ArrayPtr r = ArrayPtrFactory.empty(new long[]{1}, Sizeof.FLOAT);
        Pointer parameters = Pointer.to(Pointer.to(new int[]{a.getSize()}), a.toPTR(), r.toPTR());
        GPUNode.execute(functions.get("sum"), parameters, 1, 1, 1, 512, 1, 1, 512 * Sizeof.FLOAT);
        return r.toCPU().getFloat(0);
    }

    @Override
    public void sub(ArrayPtr a1, ArrayPtr a2) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{a1.getSize()}), a1.toPTR(), a2.toPTR());
        GPUNode.execute(functions.get("sub_array"), parameters, a1.getSize(), 1, 1, 1, 1, 1, 0);
    }

    @Override
    public void pow(ArrayPtr a1, int n) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{a1.getSize()}), a1.toPTR(), Pointer.to(new int[]{n}));
        GPUNode.execute(functions.get("pow_array"), parameters, a1.getSize(), 1, 1, 1, 1, 1, 0);
    }

    @Override
    public void mul(ArrayPtr a1, ArrayPtr a2) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{a1.getSize()}), a1.toPTR(), a2.toPTR());
        GPUNode.execute(functions.get("mul_array"), parameters, a1.getSize(), 1, 1, 1, 1, 1, 0);
    }

    @Override
    public void mul(ArrayPtr a1, float n) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{a1.getSize()}), a1.toPTR(), Pointer.to(new float[]{n}));
        GPUNode.execute(functions.get("mul_float"), parameters, a1.getSize(), 1, 1, 1, 1, 1, 0);
    }

    @Override
    public void add(ArrayPtr a1, ArrayPtr a2) {
        Pointer parameters = Pointer.to(Pointer.to(new int[]{a1.getSize()}), a1.toPTR(), a2.toPTR());
        GPUNode.execute(functions.get("add_array"), parameters, a1.getSize(), 1, 1, 1, 1, 1, 0);
    }
}
