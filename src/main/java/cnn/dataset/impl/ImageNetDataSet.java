package cnn.dataset.impl;

import cnn.data.ArrayPtr;
import cnn.data.ArrayPtrFactory;
import cnn.dataset.DataSet;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.Arrays;
import java.util.Objects;

/**
 * This class can be used to load the image net data set.
 *
 * The data set must respect the following architecture of directories and files:
 *
 * root_directory
 *        |
 *        |--> label_name_1/
 *        |        |--> image_1.png
 *        |        |--> ...
 *        |        |--> image_I.png
 *        |
 *        |--> ...
 *        |
 *        |--> label_name_L/
 *                 |--> image_1.png
 *                 |--> ...
 *                 |--> image_I.png
 *
 * The above architecture will be loaded such as the vector of label contains L elements.
 */
public class ImageNetDataSet extends DataSet {

    /**
     * Data set iterator.
     */
    private class Iterator {

        private String directory;
        private String[] labelsDir;

        private int labelsIdx;
        private int imagesIdx;

        private ArrayPtr features;
        private ArrayPtr labels;

        private int[] nbImagesPerLabel;
        private int nbImages;

        private int nbClasses;
        private int[] inputShape;

        private NativeImageLoader loader;

        /**
         * Constructor.
         * @param inputShape the shape of the input.
         * @param directory the data set directory.
         */
        private Iterator(int[] inputShape, String directory) {
            this.inputShape = inputShape;
            this.directory = directory;
            this.labelsIdx = 0;
            this.imagesIdx = 0;
            this.features = null;
            this.labels = null;
            this.loader = new NativeImageLoader(inputShape[2], inputShape[3], inputShape[1]);
        }

        /**
         * Increase the indexes of the iterator (next image).
         */
        private void incrIndexes() {
            imagesIdx++;
            if (imagesIdx >= nbImagesPerLabel[labelsIdx]) {
                imagesIdx = 0;
                labelsIdx++;
            }
            if (labelsIdx >= labelsDir.length) {
                labelsIdx = 0;
            }
        }

        /**
         * Reload data set.
         */
        private void reload() {
            labelsDir = Arrays
                    .stream(Objects.requireNonNull(new File(directory).listFiles(File::isDirectory)))
                    .map(File::toString)
                    .toArray(String[]::new);
            nbClasses = labelsDir.length;
            nbImagesPerLabel = new int[labelsDir.length];
            nbImages = 0;
            for (int i = 0; i < labelsDir.length; i++) {
                File dir = new File(labelsDir[i]);
                nbImagesPerLabel[i] = Objects.requireNonNull(dir.listFiles(File::isFile)).length;
                nbImages += nbImagesPerLabel[i];
            }
            labelsIdx = 0;
            imagesIdx = 0;
        }

        /**
         * Change the current batch, i.e. load the next batch of features and labels.
         */
        private void nextBatch() {
            if (features == null)
                features = ArrayPtrFactory.fromData(Nd4j.zeros(inputShape));
            if (labels == null)
                labels = ArrayPtrFactory.fromData(Nd4j.zeros(new int[]{inputShape[0], labelsDir.length}));
            for (int i = 0; i < inputShape[0]; i++) {
                try {
                    File[] files = Objects.requireNonNull(new File(directory + "/" + labelsDir[labelsIdx]).listFiles(File::isFile));
                    INDArray image = loader.asMatrix(files[imagesIdx]);
                    features.toCPU().put(
                            new INDArrayIndex[] {
                                    NDArrayIndex.interval(i, i + 1), NDArrayIndex.all(),
                                    NDArrayIndex.all(), NDArrayIndex.all(),
                            },
                            image
                    );
                    for (int j = 0; j < labels.getSize(); j++) {
                        labels.toCPU().putScalar(i, j, (j != labelsIdx) ? 0 : 1);
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                incrIndexes();
            }
        }

        /**
         * Check if there is a remaining batch.
         * @return true if another bath can be loaded and false otherwise.
         */
        private boolean hasNextBatch() {
            int sum = 0;
            for (int i = 0; i < labelsIdx; i++) {
                sum += nbImagesPerLabel[i];
            }
            sum += imagesIdx;
            return sum + inputShape[0] < nbImages;
        }
    }

    private Iterator training;
    private Iterator testing;

    /**
     * Load the image data set.
     * @param inputShape the input shape, i.e. [#batches, #channels, #rows, #cols].
     * @param testingDirectory the directory containing the testing images.
     * @param trainingDirectory the directory containing the training images.
     */
    public ImageNetDataSet(int[] inputShape, String testingDirectory, String trainingDirectory) {
        if (inputShape[0] < 0) {
            System.err.println("The number of batches must be positive.");
            return;
        }
        if (inputShape[1] != 3) {
            System.err.println("This data set only support RBG images, i.e. #channels must be equals to 3.");
            return;
        }
        if (inputShape[2] < 0 || inputShape[3] < 0) {
            System.err.println("The number of rows and columns must be positive.");
            return;
        }
        this.testing = new Iterator(inputShape, testingDirectory);
        testing.reload();
        this.training = new Iterator(inputShape, trainingDirectory);
        training.reload();
    }

    public ImageNetDataSet(int batchSize) {
        this(new int[]{batchSize, 3, 224, 224}, "/tmp/tmac2", "/tmp/tmac2");
    }

    /**
     * Reload both training and testing data sets.
     */
    @Override
    public void reload() {
        training.reload();
        testing.reload();
    }

    /**
     * Reload both training and testing data sets.
     * @param training true if the training data set should be reloaded or false otherwise (i.e. testing).
     */
    @Override
    public void reload(boolean training) {
        if (training)
            this.training.reload();
        else
            this.testing.reload();
    }

    /**
     * Check if there is a next batch.
     * @param training the target data set (training or testing).
     * @return true if one more batch can be loaded and false otherwise.
     */
    @Override
    public boolean hasNextBatch(boolean training) {
        return (training ? this.training : this.testing).hasNextBatch();
    }

    /**
     * Load the next batch.
     * @param training true if training data set is required and false otherwise.
     */
    @Override
    public void nextBatch(boolean training) {
        (training ? this.training : this.testing).nextBatch();
    }

    /**
     * Getter.
     * @return The number of classes in the data set.
     */
    @Override
    public int getNumberOfClasses() {
        return training.nbClasses;
    }

    /**
     * Return the images corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the images, i.e. array of shape [batchSize, #channels, #rows, #columns].
     */
    @Override
    public ArrayPtr getFeatures(boolean training) {
        return (training ? this.training : this.testing).features;
    }

    /**
     * Return the one hot encoded labels corresponding to the current batch.
     * @param training true if training data set is required and false otherwise.
     * @return the labels, i.e. array of shape [batchSize, #classes].
     */
    @Override
    public ArrayPtr getLabels(boolean training) {
        return (training ? this.training : this.testing).labels;
    }
}
