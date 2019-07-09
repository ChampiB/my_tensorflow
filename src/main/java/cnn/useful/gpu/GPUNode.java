package cnn.useful.gpu;

import cnn.nodes.Node;
import jcuda.Pointer;
import jcuda.driver.*;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;

import static jcuda.driver.JCudaDriver.*;

public abstract class GPUNode extends Node {

    static public String NODES_PATH = "./src/main/java/cnn/nodes/impl/gpu/kernel/";
    static public String OPS_PATH    = "./src/main/java/cnn/ops/gpu/kernel/";

    private Map<String, CUfunction> functions;

    /**
     * Task constructor.
     * @param kernelName the name of the kernel to load.
     * @param functionNames the names of the functions to load.
     */
    public GPUNode(String kernelPath, String kernelName, String[] functionNames) {
        functions = loadFunctions(kernelPath, kernelName, functionNames);
    }

    /**
     * Load the functions
     * @param kPath the kernels path.
     * @param kName the kernel's name.
     * @param functionNames the functions' name to load.
     * @return the map between
     */
    public static Map<String, CUfunction> loadFunctions(String kPath, String kName, String[] functionNames) {
        Map<String, CUfunction> functions = new LinkedHashMap<>();
        // Initialize the driver and create a context for the first device.
        ContextHelper.initContext();
        // Create the PTX file by calling the NVCC.
        String ptxFileName = preparePtxFile(kPath + kName);
        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        // Load all functions.
        for (String functionName : functionNames) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, functionName);
            functions.put(functionName, function);
        }
        return functions;
    }

    /**
     * Execute the kernel function.
     * @param function the function to execute.
     * @param parameters the parameters.
     * @param gridSizeX the number of blocks in dimension x.
     * @param gridSizeY the number of blocks in dimension y.
     * @param gridSizeZ the number of blocks in dimension y.
     * @param blockSizeX the number of threads in dimension x.
     * @param blockSizeY the number of threads in dimension y.
     * @param blockSizeZ the number of threads in dimension y.
     * @param sharedMemorySize the size of the shared memory.
     */
    public static void execute(
            CUfunction function, Pointer parameters,
            int gridSizeX, int gridSizeY, int gridSizeZ,
            int blockSizeX, int blockSizeY, int blockSizeZ,
            int sharedMemorySize
    ) {
        cuLaunchKernel(
                function,
                gridSizeX,  gridSizeY, gridSizeZ,
                blockSizeX, blockSizeY, blockSizeZ,
                sharedMemorySize, null,
                parameters, null
        );
        cuCtxSynchronize();
    }

    /**
     * Execute the kernel function.
     * @param name the function's name.
     * @param parameters the parameters.
     * @param blockSize the number of threads.
     * @param gridSize the number of blocks.
     * @param sharedMemorySize the size of the shared memory.
     */
    protected void execute(String name, Pointer parameters, int gridSize, int blockSize, int sharedMemorySize) {
        cuLaunchKernel(
                functions.get(name),
                gridSize,  1, 1,
                blockSize, 1, 1,
                sharedMemorySize, null,
                parameters, null
        );
        cuCtxSynchronize();
    }

    /**
     * Execute the kernel function.
     * @param name the function's name.
     * @param parameters the parameters.
     * @param gridSizeX the number of blocks in dimension x.
     * @param gridSizeY the number of blocks in dimension y.
     * @param gridSizeZ the number of blocks in dimension y.
     * @param blockSizeX the number of threads in dimension x.
     * @param blockSizeY the number of threads in dimension y.
     * @param blockSizeZ the number of threads in dimension y.
     * @param sharedMemorySize the size of the shared memory.
     */
    protected void execute(
            String name, Pointer parameters,
            int gridSizeX, int gridSizeY, int gridSizeZ,
            int blockSizeX, int blockSizeY, int blockSizeZ,
            int sharedMemorySize
    ) {
        cuLaunchKernel(
                functions.get(name),
                gridSizeX,  gridSizeY, gridSizeZ,
                blockSizeX, blockSizeY, blockSizeZ,
                sharedMemorySize, null,
                parameters, null
        );
        cuCtxSynchronize();
    }

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     */
    private static String preparePtxFile(String cuFileName)
    {
        String ptxFileName;
        try {
            int endIndex = cuFileName.lastIndexOf('.');
            if (endIndex == -1)
            {
                endIndex = cuFileName.length()-1;
            }
            ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
            File ptxFile = new File(ptxFileName);
            if (ptxFile.exists())
            {
                return ptxFileName;
            }

            File cuFile = new File(cuFileName);
            if (!cuFile.exists())
            {
                throw new IOException("Input file not found: "+cuFileName);
            }
            String modelString = "-m"+System.getProperty("sun.arch.data.model");
            String command = "nvcc " + modelString + " -ptx "+ cuFile.getPath()+" -o "+ptxFileName;
            System.out.println("Executing\n"+command);
            Process process = Runtime.getRuntime().exec(command);
            String errorMessage = new String(toByteArray(process.getErrorStream()));
            String outputMessage = new String(toByteArray(process.getInputStream()));
            int exitValue;
            try {
                exitValue = process.waitFor();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while waiting for nvcc output", e);
            }
            if (exitValue != 0)
            {
                System.out.println("nvcc process exitValue "+exitValue);
                System.out.println("errorMessage:\n"+errorMessage);
                System.out.println("outputMessage:\n"+outputMessage);
                throw new IOException("Could not create .ptx file: "+errorMessage);
            }
            System.out.println("Finished creating PTX file");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
}
