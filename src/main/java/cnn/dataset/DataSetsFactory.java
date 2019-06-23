package cnn.dataset;

import cnn.layers.LayersFactory;

public class DataSetsFactory {
    /**
     * Load an instance of data set.
     * @param name the data set name.
     * @param batchSize the batch's size.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private DataSet loadInstance(String name, int batchSize) throws Exception {
        return (DataSet) LayersFactory.class.getClassLoader()
                .loadClass("cnn.dataset.impl." + name + "DataSet")
                .getDeclaredConstructor(int.class)
                .newInstance(batchSize);
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
}
