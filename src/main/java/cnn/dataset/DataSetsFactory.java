package cnn.dataset;

import cnn.nodes.NodesFactory;

public class DataSetsFactory {
    /**
     * Load an instance of data set.
     * @param name the data set name.
     * @param batchSize the batch's size.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private DataSet loadInstance(String name, int batchSize) throws Exception {
        return (DataSet) NodesFactory.class.getClassLoader()
                .loadClass("cnn.dataset.impl." + name + "DataSet")
                .getDeclaredConstructor(int.class)
                .newInstance(batchSize);
    }

    /**
     * Load an instance of data set.
     * @param name the data set name.
     * @param batchSize the batch's size.
     * @param nb_examples the maximum number of examples to use during the training.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private DataSet loadInstance(String name, int batchSize, int nb_examples) throws Exception {
        return (DataSet) NodesFactory.class.getClassLoader()
                .loadClass("cnn.dataset.impl." + name + "DataSet")
                .getDeclaredConstructor(int.class, int.class)
                .newInstance(batchSize, nb_examples);
    }

    /**
     * Load an instance of data set.
     * @param name the data set name.
     * @param batchSize the batch's size.
     * @return the data set.
     */
    static public DataSet create(String name, int batchSize) {
        try {
            return loadInstance(name, batchSize);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Load an instance of data set.
     * @param name the data set name.
     * @param batchSize the batch's size.
     * @param nb_examples the maximum number of examples to use during the training.
     * @return the data set.
     */
    static public DataSet create(String name, int batchSize, int nb_examples) {
        try {
            return loadInstance(name, batchSize, nb_examples);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
