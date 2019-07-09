package cnn.zoo;

import cnn.graphs.Graph;

public class ModelFactory {

    /**
     * Create a model from the name.
     * @param name the model's name.
     * @param nbClasses the number of classes.
     * @return the model's graph.
     */
    static public Graph create(String name, int nbClasses) {
        try {
            return (Graph) ModelFactory.class.getClassLoader()
                    .loadClass("cnn.zoo.impl." + name)
                    .getMethod("create", int.class)
                    .invoke(null, nbClasses);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
