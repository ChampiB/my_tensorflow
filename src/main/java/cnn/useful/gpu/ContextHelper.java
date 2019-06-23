package cnn.useful.gpu;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.JCudaDriver.*;

public class ContextHelper {
    private static boolean first = true;
    public static void initContext() {
        if (first) {
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);
            first = false;
        }
    }
}
