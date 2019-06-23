# Convolutional Neural Network with CPCA and kWTA

## Introduction

This repository contains research that aims to improve Convolutional Neural Network (CNN) using a mixture of task and model learning. The idea is to update the weights based on the data (using CPCA) as well as adding competition between the neurons of the convolutional layer (using kWTA). The Conditional Principal Component Analysis (CPCA) is an Hebbian learning technique. The CPCA learning rule force each weight to tend to represent the probability of the i-th input being active given that the j-th neuron trigger, i.e. w<sub>ij</sub> = P(x<sub>i</sub> = 1 | y<sub>j</sub> = 1). This method relies on a conditioning function that allows each neuron to learn from only a subset of the data. The conditioning function function used in this work is the K-Winner-Take-All function that only allows the k most active neurons to trigger for each input.

## Framework features

### Neural networks

The framework allows to create sequential neural networks, i.e. any sequence of layers, as follow:

    // Create a neural network with four layers.
    NeuralNetwork network = new NeuralNetwork()
        .addLayer("Conv2d")
        .addLayer("MaxPooling2d")
        .addLayer("Flatten")
        .addLayer("Dense");

It is also possible to use layers' configuration to customize the layers, as follow:

    // Create a dense layer with 42 output units.
    DenseConf conf = new DenseConf(42);
    network.addLayer("Dense", new DenseConf(10));

    // Create a convolutional layer with 32 filters of size 3x3 and strides of 1x1.
    Conv2dConf conf = new Conv2dConf()
        .setFilters(new int[]{32, 3, 3})
        .setStrides(new int[]{1, 1});
    network.addLayer("Conv2d", conf);

### Layers

The framework currently support the following layers:
* **Dense**, i.e. fully connected;
* **Conv2d** (with kWTA and CPCA), i.e. apply 2d convolution;
* **MaxPooling2d**, i.e. apply 2d max pooling;
* **Flatten**, i.e. flatten the data.

Each layer has been implemented for both CPU and GPU. The correct implementation is automatically loaded depending on the device available on the computer.
