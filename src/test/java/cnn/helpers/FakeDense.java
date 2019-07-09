package cnn.helpers;

import cnn.data.ArrayPtr;
import cnn.nodes.enumerations.ActivationType;
import cnn.nodes.impl.cpu.Dense;

public class FakeDense extends Dense {

    private double delta = 0;
    private int[] pos = new int[]{0, 0};

    /**
     * Constructor.
     * @param outputSize the size of the output.
     */
    public FakeDense(int outputSize) {
        super(outputSize);
    }

    /**
     * Constructor.
     * @param outputSize the size of the output.
     * @param af the activation function.
     */
    public FakeDense(int outputSize, ActivationType af) {
        super(outputSize, af);
    }

    /**
     * Overide activation of dense layer to allow the modification of the input value on the fly.
     * @param training true if training mode and false otherwise.
     * @param x the inputs.
     * @return the activation.
     */
    @Override
    public ArrayPtr activation(boolean training, ArrayPtr... x) {
        x[0].toCPU().putScalar(pos, x[0].toCPU().getDouble(pos) + delta);
        ArrayPtr a = super.activation(training, x);
        x[0].toCPU().putScalar(pos, x[0].toCPU().getDouble(pos) - delta);
        return a;
    }

    /**
     * Setter.
     * @param delta the new value of delta.
     */
    public void setDelta(double delta) {
        this.delta = delta;
    }

    /**
     * Setter.
     * @param pos the new value of the position.
     */
    public void setPos(int[] pos) {
        this.pos = pos;
    }
}
