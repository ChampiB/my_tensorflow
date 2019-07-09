# Convolutional Neural Network with CPCA and kWTA

## Introduction

This repository contains research that aims to improve Convolutional Neural Network (CNN) using a mixture of task and model learning. The idea is to update the weights based on the data (using CPCA) as well as adding competition between the neurons of the convolutional node (using kWTA). The Conditional Principal Component Analysis (CPCA) is an Hebbian learning technique. The CPCA learning rule force each weight to tend to represent the probability of the i-th input being active given that the j-th neuron trigger, i.e. _w<sub>ij</sub> = P(x<sub>i</sub> = 1 | y<sub>j</sub> = 1)_. This method relies on a conditioning function that allows each neuron to learn from only a subset of the data. The conditioning function function used in this work is the K-Winner-Take-All function that only allows the k most active neurons to trigger for each input.

The next section presents the framework built especially for this project. The framework' structure is very similar to the structure of TensorFlow and have been re-implemented for pedagogical reasons.

## Framework features

### Graphs

The framework allows to create graphs of computation. Each graph is composed of nodes that transform the data. Some nodes contain parameters that can be trained to optimize functions, which is handy for machine learning applications.

#### Sequential neural networks

The framework allows to create sequential neural networks, i.e. any sequence of nodes, as follow:

    // Create a neural network with four nodes.
    NeuralNetwork network = new NeuralNetwork()
        .addLayer("Conv2d")
        .addLayer("MaxPooling2d")
        .addLayer("Flatten")
        .addLayer("Dense");

It is also possible to use nodes' configuration to customize the nodes, as follow:

    // Create a dense node with 42 output units.
    DenseConf conf = new DenseConf(42);
    network.addLayer("Dense", new DenseConf(10));

    // Create a convolutional node with 32 filters of size 3x3 and strides of 1x1.
    Conv2dConf conf = new Conv2dConf()
        .setFilters(new int[]{32, 3, 3})
        .setStrides(new int[]{1, 1});
    network.addLayer("Conv2d", conf);

#### General graphs of computation

Sometimes sequential models are not flexible enough and more complex graphs have to be built.

__Step 1:__ Nodes creation and connectivity.

The first step is to create the nodes and set their inputs to create the graph's connectivity.

    // Create two nodes.
    Node conv2d = NodesFactory.create("Conv2d");
    Node output = NodesFactory.create("Dense");

    // Create the connectivity.
    output.setInputs(conv2d);

__Step 2:__ Creation of the graph's configuration.

The only thing required for the creation of the configuration is to set the outputs of the graph.

    // Create the graph's configuration.
    GraphConf conf = new GraphConf().setOutputs(output);

__Step 3:__ Creation of the graph.

Once the graph's configuration is complete, creating a new graph is trivial.

    // Create the graph according to the configuration.
    Graph graph = new Graph(conf);

### Nodes

The framework currently support the following nodes:
* **Dense**, i.e. fully connected layer;
* **Conv2d** (with kWTA and CPCA), i.e. 2d convolution layer;
* **MaxPooling2d**, i.e. 2d max pooling layer;
* **Merge2d**, i.e. merge many inputs into a single output;
* **Flatten**, i.e. flatten the data;
* **Activation**, i.e. allows to apply various activation functions such as sigmoid and softmax;
* **Add**, i.e. perform an element wise addition;
* **AvgPooling2d**, i.e. 2d average pooling layer;
* **Identity**, i.e. identity node;
* **KWTA2d**, i.e. 2d K-Winners-Takes-All competition;
* **Pad2d**, i.e. 2d padding.

Each node has been implemented for both CPU and GPU. The correct implementation is automatically loaded depending on the device available on the computer.

### Zoo

The zoo contains the implementation of the following models out of the box:
* **AlexNet**, i.e. GPU implementation of a deep CNN presented by Alex Krizhevsky and al.;
* **ResNet[18|34|50|101|152]**, i.e. various stacks of residual blocks presented by Kaiming He and al.;
* **VGG[11|13|16|19]**, i.e. various very deep CNN presented by Karen Simonyan and Andrew Zisserman.

### Data sets

The framework supports the following data sets:
* **MNIST**, i.e. handwritten digits recognition;
* **TinyImageNet**, i.e. image classification with 200 classes and 500 images per class.
* **ImageNet**, i.e. most popular benchmark in image classification.
