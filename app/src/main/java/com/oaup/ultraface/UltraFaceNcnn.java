package com.oaup.ultraface;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class UltraFaceNcnn {
    public native boolean Init(AssetManager mgr);

    public native Bitmap Detect(Bitmap bitmap, boolean use_gpu);

    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("facencnn");
    }
}
