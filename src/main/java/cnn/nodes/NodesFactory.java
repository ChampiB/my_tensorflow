package cnn.nodes;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;

public class NodesFactory {
    /**
     * Force the use of a specific implementation.
     * @param type the type of implementation.
     */
    static public void forceImplementation(String type) {
        NodesFactory.type = type;
    }
    static private String type = null;

    /**
     * Load an instance of layer CPU or GPU.
     * @param name the layer name.
     * @param type the type, i.e. cpu or gpu.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private Node loadInstance(String name, String type) throws Exception {
        if (NodesFactory.type != null) {
            type = NodesFactory.type;
        }
        return (Node) NodesFactory.class.getClassLoader()
                .loadClass("cnn.nodes.impl." + type + "." + name)
                .getConstructor().newInstance();
    }

    /**
     * Load an instance of layer CPU or GPU.
     * @param name the layer name.
     * @param type the type, i.e. cpu or gpu.
     * @param object constructor's parameters.
     * @return the instance.
     * @throws Exception if an error occurred.
     */
    static private Node loadInstance(String name, String type, Object object) throws Exception {
        if (NodesFactory.type != null) {
            type = NodesFactory.type;
        }
        return (Node) NodesFactory.class.getClassLoader()
                .loadClass("cnn.nodes.impl." + type + "." + name)
                .getConstructor(Object.class).newInstance(object);
    }

    /**
     * Create an instance of the tasks.
     * If gpu is available this load the gpu version otherwise the cpu version is loaded.
     * @param name the task's name.
     * @return the task.
     */
    static public Node create(String name) {
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
     * @param object constructor's parameters.
     * @return the task.
     */
    static public Node create(String name, Object object) {
        try {
            try {
                return loadInstance(name, "gpu", object);
            } catch (Exception e) {
                return loadInstance(name, "cpu", object);
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
    static public Node create(String name, String type) {
        try {
            return loadInstance(name, type);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create an instance of the tasks.
     * If gpu is available this load the gpu version otherwise the cpu version is loaded.
     * @param name the task's name.
     * @param type the type, i.e. cpu or gpu.
     * @param object constructor's parameters.
     * @return the task.
     */
    static public Node create(String name, String type, Object object) {
        try {
            return loadInstance(name, type, object);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create an instance of the tasks.
     * If gpu is available this load the gpu version otherwise the cpu version is loaded.
     * @param kryo the task's name.
     * @param input the type, i.e. cpu or gpu.
     * @return the task.
     */
    static public Node create(Kryo kryo, Input input) {
        String name = kryo.readObject(input, String.class);
        return (name != null) ? create(name).load(kryo, input) : null;
    }
}
