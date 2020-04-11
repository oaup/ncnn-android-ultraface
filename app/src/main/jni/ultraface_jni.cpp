
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

#include "opencv2/opencv.hpp"

#include "squeezenet_v1.1.id.h"
#include "UltraFace.h"

#ifndef eprintf
#define eprintf(...) __android_log_print(ANDROID_LOG_ERROR,"@",__VA_ARGS__)
#endif

#define RGBA_A(p) (((p) & 0xFF000000) >> 24)
#define RGBA_R(p) (((p) & 0x00FF0000) >> 16)
#define RGBA_G(p) (((p) & 0x0000FF00) >>  8)
#define RGBA_B(p)  ((p) & 0x000000FF)
#define MAKE_RGBA(r,g,b,a) (((a) << 24) | ((r) << 16) | ((g) << 8) | (b))

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static std::vector<std::string> facenet_words;
static ncnn::Net facedetectnet;

static UltraFace ultraFace;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

jobject matToBitmap(JNIEnv *env, cv::Mat &srcMat){
    eprintf("mat channels: %d\n", srcMat.channels());
    if(srcMat.channels() != 3 && srcMat.channels() != 1){
        eprintf("invalid mat channels\n");
        return NULL;
    }
//create bitmap
    jclass bitmapClass = (jclass)env->FindClass("android/graphics/Bitmap");
    jmethodID bitmapCreate = env->GetStaticMethodID(bitmapClass, "createBitmap",
                                                    "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID bmpClsValueOfMid = env->GetStaticMethodID(bitmapConfigClass, "valueOf",
                                                        "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;");
    /*jmethodID bitmapConfigNativeToConfig = env->GetStaticMethodID(bitmapConfigClass,
                     "nativeToConfig","(I)Landroid/graphics/Bitmap$Config;");

     jobject bitmapConfigObject = env->CallStaticObjectMethod(bitmapConfigClass,
                                  bitmapConfigNativeToConfig,5);*/
    jobject bitmapConfigObject = env->CallStaticObjectMethod(bitmapConfigClass, bmpClsValueOfMid,
                                                             env->NewStringUTF("ARGB_8888"));

    if(!bitmapClass || !bitmapCreate || !bitmapConfigClass || !bitmapConfigObject)
        return 0;
    jobject bitmapObject = env->CallStaticObjectMethod(bitmapClass, bitmapCreate,
                                                       srcMat.cols, srcMat.rows, bitmapConfigObject);
    if(!bitmapObject)
        return 0;
    void * pixels = NULL;
    AndroidBitmapInfo bmpInfo;
    memset(&bmpInfo, 0, sizeof(bmpInfo));
    AndroidBitmap_getInfo(env, bitmapObject, &bmpInfo);
    // Check format, only RGB565 & RGBA are supported
    eprintf("bitmap format: %d\n",bmpInfo.format);
    if (bmpInfo.width <= 0 || bmpInfo.height <= 0 ||
        bmpInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888){
        eprintf("invalid bitmap\n");
        eprintf("bitmap format failed: %d\n",bmpInfo.format);
        return NULL;
    }
    int nRes = AndroidBitmap_lockPixels(env, bitmapObject, &pixels);
    if(pixels == NULL){
        eprintf("fail to lock bitmap: %d\n", nRes);
        return NULL;
    }
    int imageWidth = srcMat.cols;
    int imageHeight = srcMat.rows;
    for (int y = 0; y<imageHeight; y++) {
        uint32_t *bitmapLine = ((uint32_t *)pixels) + y * imageWidth;
        cv::Vec3b* pMatLine3 = srcMat.ptr<cv::Vec3b>(y);
        uchar* pMatLLine1 = srcMat.ptr<uchar>(y);
        for (int x = 0; x<imageWidth; x++) {
            if(srcMat.channels()==3){
                cv::Vec3b &curPixel = pMatLine3[x];
                *(bitmapLine + x) = MAKE_RGBA(curPixel[0],curPixel[1],curPixel[2],255);
            }else{
                *(bitmapLine + x) = MAKE_RGBA(pMatLLine1[x],pMatLLine1[x],pMatLLine1[x],255);
            }
        }
    }
    //end create bitmap
    AndroidBitmap_unlockPixels(env, bitmapObject);
    return bitmapObject;
}
int bitmapToMat(JNIEnv *env,jobject srcBitMap, cv::Mat& resultMat){
    // Lock the bitmap to get the buffer
    void * pixels = NULL;
    int imageWidth,imageHeight;
    AndroidBitmapInfo bmpInfo;
    memset(&bmpInfo, 0, sizeof(bmpInfo));
    AndroidBitmap_getInfo(env, srcBitMap, &bmpInfo);
    // Check format, only RGB565 & RGBA are supported
    if (bmpInfo.width <= 0 || bmpInfo.height <= 0 ||
        bmpInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        eprintf("invalid bitmap\n");
        return -1;
    }
    int nRes = AndroidBitmap_lockPixels(env, srcBitMap, &pixels);
    if(pixels == NULL){
        eprintf("fail to lock bitmap: %d\n", nRes);
        return -1;
    }
    imageWidth = bmpInfo.width;
    imageHeight = bmpInfo.height;
//    cv::Mat resultMat;
    resultMat.create(imageHeight, imageWidth, CV_8UC3);

    for (int y = 0; y<imageHeight; y++) {
        uint32_t *bitmapLine = ((uint32_t *)pixels) + y * imageWidth;
        cv::Vec3b* pMatLine3 = resultMat.ptr<cv::Vec3b>(y);
        for (int x = 0; x<imageWidth; x++) {
            uint32_t v = *(bitmapLine + x);
            //int aValue = RGBA_A(v);
            // eprintf("print lock A: %d\n", aValue);

            //if(aValue != 255)
            // eprintf("A: %d at (%d, %d) \n", aValue, x, y);

            cv::Vec3b &curPixel = pMatLine3[x];
            curPixel[2] = RGBA_B(v);
            curPixel[1] = RGBA_G(v);
            curPixel[0] = RGBA_R(v);
            //curPixel[3] = RGBA_A(v);
        }
    }
    // cv::imwrite("/mnt/sdcard/opencv/ConvertedRGB.txt", );
    AndroidBitmap_unlockPixels(env, srcBitMap);
    return 0;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "FaceNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "FaceNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL Java_com_oaup_ultraface_UltraFaceNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager) {
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

   AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

   facedetectnet.opt = opt;

    // init param
    {
        int ret = facedetectnet.load_param(mgr, "slim_320.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "FaceNcnn", "load_param_bin failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = facedetectnet.load_model(mgr, "slim_320.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    ultraFace.init(&facedetectnet,640, 480, 1, 0.6);

    // init words
    {
        AAsset* asset = AAssetManager_open(mgr, "synset_words.txt", AASSET_MODE_BUFFER);
        if (!asset)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "open synset_words.txt failed");
            return JNI_FALSE;
        }

        int len = AAsset_getLength(asset);

        std::string words_buffer;
        words_buffer.resize(len);
        int ret = AAsset_read(asset, (void*)words_buffer.data(), len);

        AAsset_close(asset);

        if (ret != len)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "read synset_words.txt failed");
            return JNI_FALSE;
        }

        facenet_words = split_string(words_buffer, "\n");
    }

    return JNI_TRUE;
}

// public native String Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobject JNICALL Java_com_oaup_ultraface_UltraFaceNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    //if (width != 227 || height != 227)
    //return NULL;

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;
    // opencv mat
    cv::Mat frame;
    int ret = bitmapToMat(env, bitmap, frame);
    if(ret != 0){
        return NULL;
    }
    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR);
    // face detector
    std::vector<FaceInfo> face_info;
    ultraFace.detect(in,face_info);

    for (int i = 0; i < face_info.size(); i++) {
        auto face = face_info[i];
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);
        cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }
    std::string text = std::to_string(face_info.size());
    cv::putText(frame,text,cv::Point(30, 30),cv::FONT_HERSHEY_SIMPLEX, 0.7,cv::Scalar(0, 0, 255), 2);

    return matToBitmap(env, frame);
}

}