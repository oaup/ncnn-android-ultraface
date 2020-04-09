LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
# change this folder path to yours
NCNN_INSTALL_PATH := E:/Android/ncnn-android-vulkan-lib

LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := $(NCNN_INSTALL_PATH)/$(TARGET_ARCH_ABI)/libncnn.a

include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

# change OpenCV-android-sdk folder path to yours
OPENCVROOT := E:/Android/library/OpenCV-android-sdk
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES := ultraface_jni.cpp UltraFace.cpp

LOCAL_C_INCLUDES := $(NCNN_INSTALL_PATH)/include $(NCNN_INSTALL_PATH)/include/ncnn $(OPENCVROOT)/sdk/native/jni/include

LOCAL_STATIC_LIBRARIES := ncnn

LOCAL_CFLAGS := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_LDFLAGS += -Wl,--gc-sections

LOCAL_CFLAGS += -fopenmp
LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp

LOCAL_LDLIBS += -lz -llog -ljnigraphics -lvulkan -landroid
LOCAL_MODULE := facencnn

include $(BUILD_SHARED_LIBRARY)
