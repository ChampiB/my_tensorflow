package cnn.useful.cpu;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

public class ThreadPool {

    static private ThreadPool instance = new ThreadPool();

    static public ThreadPoolExecutor getInstance() {
        return instance.get();
    }

    private ThreadPoolExecutor executor;

    private ThreadPool() {
        executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    private ThreadPoolExecutor get() {
        return executor;
    }

    /**
     * Wait for all tasks.
     * @param tasks the tasks to wait.
     */
    static public void waitAll(List<Future<Boolean>> tasks) {
        for (Future<Boolean> task: tasks) {
            try {
                task.get();
            } catch (Exception e) {
                System.err.println("Unable to complete task: " + e.getMessage());
            }
        }
    }

}
