package cnn.dataset;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;

/**
 * Helper for the MNIST data set.
 */
public class MnistDataSet {

    private int seed;
    private int batchSize;
    private int nRows;
    private int nCols;
    private MnistDataSetIterator train;
    private MnistDataSetIterator test;

    /**
     * Create the helper for the MNIST data set.
     * @param seed for reproducibility.
     * @param batchSize is the number of examples per batch
     * @param nRows is the number of rows per image
     * @param nCols is the number of columns per image
     */
    public MnistDataSet(int seed, int batchSize, int nRows, int nCols) {
        this.seed = seed;
        this.batchSize = batchSize;
        this.nRows = nRows;
        this.nCols = nCols;
        this.train = getTrainingIterator();
        this.test = getTestingIterator();
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
            return train.hasNext();
        } else {
            return test.hasNext();
        }
    }

    /**
     * Getter.
     * @param training defines the data set to use (training vs testing).
     * @return the next batch.
     */
    public DataSet nextBatch(boolean training) {
        if (training) {
            return train.next();
        } else {
            return test.next();
        }
    }

    /**
     * Getter.
     * @return the batches' shape.
     */
    public int[] batchShape() {
        return new int[]{batchSize, 1, nRows, nCols};
    }

    /**
     * Reload the data set.
     */
    public void reload() {
        this.train = getTrainingIterator();
        this.test = getTestingIterator();
    }
}
