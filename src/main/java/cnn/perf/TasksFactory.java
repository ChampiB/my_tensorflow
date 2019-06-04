package cnn.perf;

public class TasksFactory {
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
            System.err.println(e.getMessage());
        }
        return null;
    }
}
