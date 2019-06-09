package cnn.layers.perf;

public class TasksFactory {
    /**
     * Force the use of a specific implementation.
     * @param type the type of implementation.
     */
    static public void forceImplementation(String type) {
        TasksFactory.type = type;
    }
    static private String type = null;

    /**
     * Load an instance of layer CPU or GPU.
     * @param name the layer name.
     * @param type the type, i.e. cpu or gpu.
     * @param <T> the type of class.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private<T> T loadInstance(String name, String type) throws Exception {
        if (TasksFactory.type != null) {
            type = TasksFactory.type;
        }
        return (T) TasksFactory.class.getClassLoader()
                .loadClass("cnn.layers.perf." + type + "." + name)
                .getConstructor().newInstance();
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
                return loadInstance(name, "gpu");
            } catch (Exception e) {
                return loadInstance(name, "cpu");
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
            return loadInstance(name, type);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
