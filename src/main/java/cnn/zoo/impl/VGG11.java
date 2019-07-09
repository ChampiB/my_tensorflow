package cnn.zoo.impl;

import cnn.graphs.Graph;
import cnn.graphs.impl.NeuralNetwork;
import cnn.nodes.conf.Conv2dConf;
import cnn.nodes.conf.DenseConf;
import cnn.nodes.enumerations.ActivationType;

public class VGG11 {
    /**
     * Create the VGG11 model.
     * @param nbClasses the number of classes, i.e. number of outputs.
     * @return the computational graph.
     */
    public static Graph create(int nbClasses) {
        return new NeuralNetwork()
                .addLayer("Conv2d", new Conv2dConf(new int[]{64, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{128, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{256, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{256, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{512, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{512, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{512, 3, 3}, new int[]{1, 1}))
                .addLayer("Conv2d", new Conv2dConf(new int[]{512, 3, 3}, new int[]{1, 1}))
                .addLayer("MaxPooling2d")
                .addLayer("Flatten")
                .addLayer("Dense", new DenseConf(4096))
                .addLayer("Dense", new DenseConf(4096))
                .addLayer("Dense", new DenseConf(nbClasses).setAf(ActivationType.SOFTMAX));
    }
}
