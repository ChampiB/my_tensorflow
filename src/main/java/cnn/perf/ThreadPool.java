package cnn.perf;

import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class ThreadPool {

    private static ThreadPool instance = new ThreadPool();

    public static ThreadPoolExecutor getInstance() {
        return instance.get();
    }

    private ThreadPoolExecutor executor;

    private ThreadPool() {
        executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    private ThreadPoolExecutor get() {
        return executor;
    }
}
