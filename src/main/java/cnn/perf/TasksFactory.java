package cnn.perf;

public class TasksFactory {
    /**
     * No GPU available.
     */
    static public class NoGPUException extends RuntimeException {
        @Override
        public String getMessage() {
            return "[Warning]: No GPU available.";
        }
    }

    /**
     * Create an instance of the tasks.
     * If gpu is available this load the gpu version otherwise the cpu version is loaded.
     * @param name the task's name.
     * @param <T> the type of task created.
     * @return the task.
     */
    static public<T> T create(String name) {
        try {
            try {
                return (T) TasksFactory.class.getClassLoader()
                        .loadClass("cnn.perf.gpu." + name)
                        .getConstructor().newInstance();
            } catch (Exception e) {
                return (T) TasksFactory.class.getClassLoader()
                        .loadClass("cnn.perf.cpu." + name)
                        .getConstructor().newInstance();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create an instance of the tasks.
     * If gpu is available this load the gpu version otherwise the cpu version is loaded.
     * @param name the task's name.
     * @param type the type, i.e. cpu or gpu.
     * @param <T> the type of task created.
     * @return the task.
     */
    static public<T> T create(String name, String type) {
        try {
            return (T) TasksFactory.class.getClassLoader()
                    .loadClass("cnn.perf." + type + "." + name)
                    .getConstructor().newInstance();
        } catch (Exception e) {
            if (type.equals("gpu"))
                throw new NoGPUException();
            else
                throw new RuntimeException(e);
        }
    }
}
