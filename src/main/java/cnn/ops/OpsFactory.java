package cnn.ops;

public class OpsFactory {

    /**
     * Load an instance of layer CPU or GPU.
     * @param name the layer name.
     * @param type the type, i.e. cpu or gpu.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private <T> T loadInstance(String name, String type) throws Exception {
        return (T) OpsFactory.class.getClassLoader()
                .loadClass("cnn.ops." + type + "." + name)
                .getConstructor().newInstance();
    }

    /**
     * Create an instance of the tasks.
     * If gpu is available this load the gpu version otherwise the cpu version is loaded.
     * @param name the task's name.
     * @return the task.
     */
    static public <T> T create(String name) {
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
     * @return the task.
     */
    static public <T> T create(String name, String type) {
        try {
            return loadInstance(name, type);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
