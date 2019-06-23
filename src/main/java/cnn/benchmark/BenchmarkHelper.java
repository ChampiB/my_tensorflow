package cnn.benchmark;

import java.util.concurrent.Callable;

public class BenchmarkHelper {
    public static double time(Callable c) {
        try {
            long start_time = System.nanoTime();
            c.call();
            long end_time = System.nanoTime();
            return (end_time - start_time) / 1e6;
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return -1;
        }
    }
}
