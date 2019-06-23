package cnn.layers;

import cnn.useful.ArrayPtr;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.BufferedReader;
import java.io.BufferedWriter;

/**
 * Generic layer abstraction.
 */
public interface Layer {
    /**
     * Compute the layer activation.
     * @param x is the input.
     * @param training the mode (training vs testing).
     * @return the activation.
     */
    ArrayPtr activation(ArrayPtr x, boolean training);

    /**
     * Update the weights.
     * @param gradient the back propagation gradient from the upper layer.
     * @param lr the learning rate.
     * @return the back propagation gradient from this layer.
     */
    ArrayPtr update(ArrayPtr gradient, double lr);

    /**
     * Save the layer to the file.
     * @param kryo the kryo object.
     * @param output the kryo output.
     */
    void save(Kryo kryo, Output output);

    /**
     * Load weights from file.
     * @param kryo the kryo object.
     * @param input the kryo input.
     */
    Layer loadWeights(Kryo kryo, Input input);

    /**
     * Load layer from file.
     * @param kryo the kryo object.
     * @param input the kryo input.
     */
    Layer load(Kryo kryo, Input input);

    /**
     * Display the layer on the standard output.
     */
    void print();
}
