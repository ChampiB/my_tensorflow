package cnn.dataset;

import cnn.useful.ArrayPtr;

public abstract class DataSet {
    /**
     * Reload the data set.
     */
    public abstract void reload();

    /**
     * Reload the data set.
     * @param training true if training data set is required and false otherwise.
     */
    public abstract void reload(boolean training);

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
    public abstract ArrayPtr getFeatures(boolean training);

    /**
     * Return the labels corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the labels.
     */
    public abstract ArrayPtr getLabels(boolean training);
}
