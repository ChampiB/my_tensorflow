package cnn.nodes.impl.cpu.useful;

import cnn.useful.cpu.ThreadPool;
import org.junit.jupiter.api.Test;

import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.jupiter.api.Assertions.*;

class ThreadPoolTest {

    @Test
    void getInstance() {
        ThreadPoolExecutor tp = ThreadPool.getInstance();
        Future<Integer> result = tp.submit(() -> 1 + 1);
        try {
            assertEquals(2, result.get(), "Task must success and return 2.");
        } catch (Exception e) {
            fail();
        }
    }
}