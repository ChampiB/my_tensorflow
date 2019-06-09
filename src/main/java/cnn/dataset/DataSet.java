package cnn.dataset;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class DataSet {
    /**
     * Reload the data set.
     */
    public abstract void reload();

    /**
     * Check if there is a next batch.
     * @param training true if training data set is required and false otherwise.
     * @return true if there is a next batch and false otherwise.
     */
    public abstract boolean hasNextBatch(boolean training);

    /**
     * Selection the next batch in the training set.
     * @param training true if training data set is required and false otherwise.
     */
    public abstract void nextBatch(boolean training);

    /**
     * Return the number of classes.
     * @return the number of classes.
     */
    public abstract int getNumberOfClasses();

    /**
     * Return the features corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the features.
     */
    public abstract INDArray getFeatures(boolean training);

    /**
     * Return the labels corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the labels.
     */
    public abstract INDArray getLabels(boolean training);
}
