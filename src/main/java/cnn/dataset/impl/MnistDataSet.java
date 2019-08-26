package cnn.dataset.impl;

import cnn.data.ArrayPtr;
import cnn.data.ArrayPtrFactory;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import java.io.IOException;

/**
 * Helper for the MNIST data set.
 */
public class MnistDataSet extends cnn.dataset.DataSet {

    private int seed;
    private int batchSize;
    private int nRows;
    private int nCols;
    private MnistDataSetIterator train;
    private MnistDataSetIterator test;
    private DataSet trainSet;
    private DataSet testSet;
    private int maxNbExamples;
    private int batchIndex;

    /**
     * Create the helper for the MNIST data set.
     * @param seed for reproducibility.
     * @param batchSize is the number of examples per batch
     * @param nRows is the number of rows per image
     * @param nCols is the number of columns per image
     */
    public MnistDataSet(int seed, int batchSize, int nRows, int nCols) {
        this.batchIndex = 0;
        this.maxNbExamples = -1;
        this.seed = seed;
        this.batchSize = batchSize;
        this.nRows = nRows;
        this.nCols = nCols;
        this.train = getTrainingIterator();
        this.test = getTestingIterator();
    }

    /**
     * Create the helper for the MNIST data set.
     * @param batchSize is the number of examples per batch
     * @param nRows is the number of rows per image
     * @param nCols is the number of columns per image
     */
    public MnistDataSet(int batchSize, int nRows, int nCols) {
        this(123, batchSize, nRows, nCols);
    }

    /**
     * Create the helper for the MNIST data set.
     * @param batchSize is the number of examples per batch
     */
    public MnistDataSet(int batchSize) {
        this(123, batchSize, 28, 28);
    }

    /**
     * Create the helper for the MNIST data set.
     * @param batchSize is the number of examples per batch
     * @param maxNbExamples the maximum number of examples to use during the training.
     */
    public MnistDataSet(int batchSize, int maxNbExamples) {
        this(123, batchSize, 28, 28);
        this.maxNbExamples = maxNbExamples;
    }

    /**
     * Return the number of classes.
     * @return the number of classes.
     */
    public int getNumberOfClasses() {
        return 10;
    }

    /**
     * Getter.
     * @return the training set iterator.
     */
    private MnistDataSetIterator getTrainingIterator() {
        try {
            return new MnistDataSetIterator(batchSize, true, seed);
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return null;
    }

    /**
     * Getter.
     * @return the testing set iterator.
     */
    private MnistDataSetIterator getTestingIterator() {
        try {
            return new MnistDataSetIterator(batchSize, false, seed);
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return null;
    }

    /**
     * Question: Is there a next batch ?
     * @return the answer.
     */
    public boolean hasNextBatch(boolean training) {
        if (training) {
            if (maxNbExamples != -1 && (batchIndex + 1) * batchSize > maxNbExamples)
                return false;
            return train.hasNext();
        } else {
            return test.hasNext();
        }
    }

    /**
     * Selection the next batch.
     * @param training defines the data set to use (training vs testing).
     */
    public void nextBatch(boolean training) {
        batchIndex++;
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
        return ArrayPtrFactory.fromData(getFeaturesArray(training));
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
        return ArrayPtrFactory.fromData(getLabelsArray(training));
    }

    /**
     * Getter.
     * @return the batches' shape.
     */
    private int[] batchShape() {
        return new int[]{batchSize, 1, nRows, nCols};
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
        batchIndex = 0;
        if (training) {
            this.train = getTrainingIterator();
        } else {
            this.test = getTestingIterator();
        }
    }
}
