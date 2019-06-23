package cnn.dataset.impl;

import cnn.useful.ArrayPtr;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TinyImageNetDataSet extends cnn.dataset.DataSet {

    private int batchSize;
    private int nRows;
    private int nCols;
    private TinyImageNetDataSetIterator train;
    private TinyImageNetDataSetIterator test;
    private org.nd4j.linalg.dataset.DataSet trainSet;
    private org.nd4j.linalg.dataset.DataSet testSet;

    /**
     * Create the helper for the TinyImageNet data set.
     * @param batchSize is the number of examples per batch
     */
    public TinyImageNetDataSet(int batchSize) {
        this.batchSize = batchSize;
        this.nRows = 64;
        this.nCols = 64;
        this.train = getTrainingIterator();
        this.test = getTestingIterator();
    }

    /**
     * Getter.
     * @return the training set iterator.
     */
    private TinyImageNetDataSetIterator getTrainingIterator() {
        try {
            return new TinyImageNetDataSetIterator(batchSize, DataSetType.TRAIN);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        return null;
    }

    /**
     * Getter.
     * @return the testing set iterator.
     */
    private TinyImageNetDataSetIterator getTestingIterator() {
        try {
            return new TinyImageNetDataSetIterator(batchSize, DataSetType.TEST);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        return null;
    }

    /**
     * Return the number of classes.
     * @return the number of classes.
     */
    public int getNumberOfClasses() {
        return 200;
    }

    /**
     * Getter.
     * @return the batches' shape.
     */
    private int[] batchShape() {
        return new int[]{batchSize, 3, nRows, nCols};
    }

    /**
     * Reload the data set.
     */
    public void reload() {
        reload(true);
        reload(false);
    }

    /**
     * Reload the data set.
     */
    public void reload(boolean training) {
        if (training) {
            this.train = getTrainingIterator();
        } else {
            this.test = getTestingIterator();
        }
    }
    /**
     * Check if there is a next batch.
     * @param training true if training data set is required and false otherwise.
     * @return true if there is a next batch and false otherwise.
     */
    public boolean hasNextBatch(boolean training) {
        if (training) {
            return train.hasNext();
        } else {
            return test.hasNext();
        }
    }

    /**
     * Selection the next batch in the training set.
     * @param training true if training data set is required and false otherwise.
     */
    public void nextBatch(boolean training) {
        if (training) {
            trainSet = train.next();
        } else {
            testSet = test.next();
        }
    }

    /**
     * Return the features corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the features.
     */
    public INDArray getFeaturesArray(boolean training) {
        if (training) {
            return trainSet.getFeatures().reshape(batchShape());
        } else {
            return testSet.getFeatures().reshape(batchShape());
        }
    }

    /**
     * Return the features corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the features.
     */
    public ArrayPtr getFeatures(boolean training) {
        return new ArrayPtr(getFeaturesArray(training));
    }

    /**
     * Return the labels corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the labels.
     */
    public INDArray getLabelsArray(boolean training) {
        if (training) {
            return trainSet.getLabels();
        } else {
            return testSet.getLabels();
        }
    }

    /**
     * Return the labels corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the labels.
     */
    public ArrayPtr getLabels(boolean training) {
        return new ArrayPtr(getLabelsArray(training));
    }
}
