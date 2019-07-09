package cnn.examples;

import cnn.graphs.impl.NeuralNetwork;
import cnn.dataset.DataSet;
import cnn.dataset.DataSetsFactory;
import cnn.nodes.conf.Conv2dConf;
import cnn.nodes.conf.DenseConf;
import cnn.useful.debug.impl.AccuracyTestingSet;
import cnn.useful.stopping.EpochsStopCondition;
import cnn.data.ArrayPtr;

import java.lang.reflect.Method;
import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.Random;

class CNN_vs_HCNN_accuracy {

    private static String file = "weights" + new Random().nextInt() + ".save";

    static class Parameters {

        String[] testParameters;
        int testId;
        int epochs;
        int debug;
        int[] kernels;

        private boolean t;
        private boolean p;
        private boolean d;
        private boolean e;
        private boolean k;

        Parameters(String[] args) {
            if (args.length != 5) throw new InvalidParameterException();
            for (String arg : args) parse(arg);
            if (!t || !p || !d || !e || !k) throw new InvalidParameterException();
        }

        private void parse(String arg) {
            String[] table = arg.split("=");
            if (table.length != 2) throw new InvalidParameterException();
            switch (table[0]) {
                case "-t":
                    t = true;
                    parseT(table[1]);
                    break;
                case "-p":
                    p = true;
                    parseP(table[1]);
                    break;
                case "-d":
                    d = true;
                    parseD(table[1]);
                    break;
                case "-e":
                    e = true;
                    parseE(table[1]);
                    break;
                case "-k":
                    k = true;
                    parseK(table[1]);
                    break;
                default:
                    throw new InvalidParameterException();
            }
        }

        void parseT(String arg) {
            testId = Integer.valueOf(arg);
        }

        void parseP(String arg) {
            testParameters = arg.split(",");
        }

        void parseD(String arg) {
            debug = Integer.valueOf(arg);
        }

        void parseE(String arg) {
            epochs = Integer.valueOf(arg);
        }

        void parseK(String arg) {
            String[] strings = arg.split(",");
            if (strings.length != 3) throw new InvalidParameterException();
            kernels = new int[strings.length];
            for (int i = 0; i < strings.length; i++) {
                kernels[i] = Integer.valueOf(strings[i]);
            }
        }
    }

    private static void network1(Parameters param) {
        DataSet dataSet = DataSetsFactory.create("Mnist", 20);
        int k = Integer.valueOf(param.testParameters[0]);
        System.out.println("k == " + k);
        NeuralNetwork cnn = new NeuralNetwork()
                .addLayer("Conv2d", new Conv2dConf(k, 0).setFilters(param.kernels))
                .addLayer("MaxPooling2d")
                .addLayer("Flatten")
                .addLayer("Dense", new DenseConf(dataSet.getNumberOfClasses()));
        cnn.loadWeights(file);
        cnn.fit(dataSet, new EpochsStopCondition(param.epochs), 0.01, new AccuracyTestingSet(param.debug));
        cnn.evaluate(dataSet);
    }

    private static void network2(Parameters param) {
        DataSet dataSet = DataSetsFactory.create("Mnist", 20);
        float ratio = Float.valueOf(param.testParameters[0]);
        System.out.println("ratio == " + ratio);
        NeuralNetwork cnn = new NeuralNetwork()
                .addLayer("Conv2d", new Conv2dConf(0, ratio).setFilters(param.kernels))
                .addLayer("MaxPooling2d")
                .addLayer("Flatten")
                .addLayer("Dense", new DenseConf(dataSet.getNumberOfClasses()));
        cnn.loadWeights(file);
        cnn.fit(dataSet, new EpochsStopCondition(param.epochs), 0.01, new AccuracyTestingSet(param.debug));
        cnn.evaluate(dataSet);
    }

    private static void network3(Parameters param) {
        DataSet dataSet = DataSetsFactory.create("Mnist", 20);
        int k = Integer.valueOf(param.testParameters[0]);
        float ratio = Float.valueOf(param.testParameters[1]);
        System.out.println("k == " + k + ", ratio == " + ratio);
        NeuralNetwork cnn = new NeuralNetwork()
                .addLayer("Conv2d", new Conv2dConf(k, ratio).setFilters(param.kernels))
                .addLayer("MaxPooling2d")
                .addLayer("Flatten")
                .addLayer("Dense", new DenseConf(dataSet.getNumberOfClasses()));
        cnn.loadWeights(file);
        cnn.fit(dataSet, new EpochsStopCondition(param.epochs), 0.01, new AccuracyTestingSet(param.debug));
        cnn.evaluate(dataSet);
    }

    private static void initWeights(DataSet dataSet, NeuralNetwork network) {
        dataSet.nextBatch(true);
        ArrayPtr features = dataSet.getFeatures(true);
        network.activation(true, features);
        dataSet.reload(true);
    }

    private static void run(Parameters param) {
        System.out.println("Start test " + param.testId + " for " + param.epochs + " epochs with debug every " + param.debug + " iterations.");
        System.out.println("Filters: " + Arrays.toString(param.kernels));
        try {
            // Baseline.
            System.out.println("Start baseline.");
            DataSet dataSet = DataSetsFactory.create("Mnist", 20);
            NeuralNetwork cnn = new NeuralNetwork()
                    .addLayer("Conv2d", new Conv2dConf().setFilters(param.kernels))
                    .addLayer("MaxPooling2d")
                    .addLayer("Flatten")
                    .addLayer("Dense", new DenseConf(dataSet.getNumberOfClasses()));
            initWeights(dataSet, cnn);
            cnn.save(file);
            cnn.fit(dataSet, new EpochsStopCondition(param.epochs), 0.01, new AccuracyTestingSet(param.debug));
            cnn.evaluate(dataSet);
            System.out.println("Done baseline.");
            // Benchmark.
            System.out.println("Start examples.");
            String functionName = "network" + param.testId;
            Method method = CNN_vs_HCNN_accuracy.class.getDeclaredMethod(functionName, Parameters.class);
            method.invoke(null, param);
            System.out.println("Done examples.");
        } catch (Exception e) {
            System.out.println("Benchmark failed.");
            throw new RuntimeException(e);
        }
        System.out.println("Done.");
    }

    private static void help() {
        System.out.println("NAME");
        System.out.println("\tCNN_vs_HCNN_accuracy - run accuracy benchmarks between CNN and HCNN");
        System.out.println();
        System.out.println("SYNOPSIS");
        System.out.println("\t./CNN_vs_HCNN_accuracy.java -t=test_id -p=test_parameters -d=debug -e=epochs -e=epochs -k=size_of_filters");
        System.out.println();
        System.out.println("DESCRIPTION");
        System.out.println();
        System.out.println("Run the examples specified by the arguments.");
        System.out.println();
        System.out.println("All arguments are mandatory:");
        System.out.println();
        System.out.println("\t-t=test_id");
        System.out.println("\t\tspecify the test id.");
        System.out.println();
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\t   | test id | layer id | layer type           |");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\t   |       1 |        1 | Conv2d + kWTA        |");
        System.out.println("\t\t   |         |        2 | MaxPooling2d         |");
        System.out.println("\t\t   |         |        3 | Flatten              |");
        System.out.println("\t\t   |         |        4 | Dense                |");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\t   |       2 |        1 | Conv2d + CPCA        |");
        System.out.println("\t\t   |         |        2 | MaxPooling2d         |");
        System.out.println("\t\t   |         |        3 | Flatten              |");
        System.out.println("\t\t   |         |        4 | Dense                |");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\t   |       3 |        1 | Conv2d + CPCA + kWTA |");
        System.out.println("\t\t   |         |        2 | MaxPooling2d         |");
        System.out.println("\t\t   |         |        3 | Flatten              |");
        System.out.println("\t\t   |         |        4 | Dense                |");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\tTable 1: table of test id and network architecture.");
        System.out.println();
        System.out.println("\t-p=test_parameters");
        System.out.println("\t\tcomma-separated list that specify the parameters of the test;");
        System.out.println("\t\tthe number of parameters depends of the network architecture;");
        System.out.println("\t\tone additional parameter is required for each layer with either kWTA or CPCA;");
        System.out.println("\t\ttwo additional parameters are required for each layer with both kWTA and CPCA.");
        System.out.println();
        System.out.println("\t\tLet's consider the following network:");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\t   | test id | layer id | layer type           |");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\t   |       1 |        1 | Conv2d + CPCA + kWTA |");
        System.out.println("\t\t   |         |        2 | Conv2d + kWTA        |");
        System.out.println("\t\t   |         |        3 | Conv2d + CPCA        |");
        System.out.println("\t\t   |         |        4 | MaxPooling2d         |");
        System.out.println("\t\t   |         |        5 | Flatten              |");
        System.out.println("\t\t   |         |        6 | Dense                |");
        System.out.println("\t\t   #---------#----------#----------------------#");
        System.out.println("\t\tThe required parameters are as follow: -p=k1,ratio1,k2,ratio3.");
        System.out.println();
        System.out.println("\t-d=debug");
        System.out.println("\t\tspecify the number of debug between each accuracy statement, i.e. any positive integer.");
        System.out.println();
        System.out.println("\t-e=epochs");
        System.out.println("\t\tspecify the number of epochs to run, i.e. any positive integer.");
        System.out.println();
        System.out.println("\t-k=size_of_filters");
        System.out.println("\t\tcomma separated list that specify the dimensions of the filters, i.e. number_of_filters,width_of_kernels,height_of_kernel.");
        System.out.println();
        System.out.println("EXAMPLE");
        System.out.println("\t./CNN_vs_HCNN_accuracy.java -t=2 -p=1 -d=100 -e=1 -k=32,2,2");
        System.out.println();
    }

    public static void main(String[] args) {
        try {
            Parameters param = new Parameters(args);
            run(param);
            System.exit(0);
        } catch (RuntimeException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            help();
            System.exit(1);
        }
    }
}
