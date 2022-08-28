FROM ubuntu:20.04

ENV DEBCONF_NOWARNINGS yes
ENV DEBIAN_FRONTEND noninteractive
# timezone setting
ENV TZ Asia/Tokyo

ARG OPENCV_VERSION='4.5.5'

# Build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    tzdata \
    git \
    cmake \
    wget \
    unzip \
    checkinstall \
    build-essential

# Media I/O
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libgdal-dev \
    libgtk2.0-dev

# Video I/O:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libopencv-highgui-dev \
    ffmpeg

# Parallelism and linear algebra libraries:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libtbb-dev \
    libeigen3-dev

# Optimization libraries for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    gfortran

# GTK and QT5 lib for the graphical user functionalites coming from OpenCV highghui module
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgtk-3-dev \
    qtbase5-dev \
    qttools5-dev-tools \
    qt5-default

# VTK
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vtk7 \
    tcl-vtk7 \
    python3-vtk7 \
    libvtk7-dev \
    libvtk7-qt-dev

# Python3:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-tk \
    python3-numpy

# Extra
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    x11-apps \
    qv4l2

# OPENCV install
WORKDIR /opt
RUN wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && \
    wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    mkdir build && \
    cd build && \
    cmake \
    -D BUILD_opencv_world=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_FORCE_3RDPARTY_BUILD=ON \
    -D BUILD_ZLIB=ON \
    -D BUILD_TIFF=ON \
    -D BUILD_JASPER=OFF \
    -D BUILD_JPEG=ON \
    -D BUILD_PNG=ON \
    -D BUILD_OPENEXR=ON \
    -D BUILD_TBB=ON \
    -D BUILD_IPP_IW=ON \
    -D BUILD_ITT=ON \
    -D WITH_1394=ON \
    -D WITH_AVFOUNDATION=ON \
    -D WITH_CAP_IOS=ON \
    -D WITH_CAROTENE=ON \
    -D WITH_CPUFEATURES=ON \
    -D WITH_VTK=ON \
    -D WITH_CUDA=OFF \
    -D WITH_CUFFT=OFF \
    -D WITH_CUBLAS=OFF \
    -D WITH_CUDNN=OFF \
    -D WITH_NVCUVID=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GTK=ON \
    -D WITH_GTK_2_X=ON \
    -D WITH_IPP=ON \
    -D WITH_HALIDE=ON \
    -D WITH_VULKAN=ON \
    -D WITH_INF_ENGINE=ON \
    -D WITH_NGRAPH=ON \
    -D WITH_JASPER=OFF \
    -D WITH_OPENJPEG=ON \
    -D WITH_JPEG=ON \
    -D WITH_WEBP=ON \
    -D WITH_OPENEXR=ON \
    -D WITH_OPENGL=ON \
    -D WITH_OPENVX=ON \
    -D WITH_OPENNI=OFF \
    -D WITH_OPENNI2=OFF \
    -D WITH_PNG=ON \
    -D WITH_GDCM=ON \
    -D WITH_PVAPI=ON \
    -D WITH_ARAVIS=ON \
    -D WITH_QT=ON \
    -D WITH_WIN32UI=ON \
    -D WITH_TBB=ON \
    -D WITH_HPX=OFF \
    -D WITH_OPENMP=ON \
    -D WITH_PTHREADS_PF=ON \
    -D WITH_TIFF=ON \
    -D WITH_V4L=ON \
    -D WITH_DSHOW=ON \
    -D WITH_MSMF=ON \
    -D WITH_MSMF_DXVA=ON \
    -D WITH_XIMEA=ON \
    -D WITH_UEYE=ON \
    -D WITH_XINE=ON \
    -D WITH_CLP=ON \
    -D WITH_OPENCL=ON \
    -D WITH_OPENCL_SVM=ON \
    -D WITH_OPENCLAMDFFT=ON \
    -D WITH_OPENCLAMDBLAS=ON \
    -D WITH_DIRECTX=ON \
    -D WITH_OPENCL_D3D11_NV=ON \
    -D WITH_LIBREALSENSE=ON \
    -D WITH_VA=ON \
    -D WITH_VA_INTEL=OFF \
    -D WITH_MFX=ON \
    -D WITH_GDAL=ON \
    -D WITH_GPHOTO2=ON \
    -D WITH_LAPACK=ON \
    -D WITH_ITT=ON \
    -D WITH_PROTOBUF=ON \
    -D WITH_IMGCODEC_HDR=ON \
    -D WITH_IMGCODEC_SUNRASTER=ON \
    -D WITH_IMGCODEC_PXM=ON \
    -D WITH_IMGCODEC_PFM=ON \
    -D WITH_QUIRC=ON \
    -D WITH_ANDROID_MEDIANDK=ON \
    -D WITH_ANDROID_NATIVE_CAMERA=ON \
    -D WITH_TENGINE=ON \
    -D WITH_ONNX=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_opencv_js=OFF \
    -D BUILD_ANDROID_PROJECTS=OFF \
    -D BUILD_ANDROID_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_PACKAGE=ON \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_WITH_DEBUG_INFO=OFF \
    -D BUILD_WITH_STATIC_CRT=OFF \
    -D BUILD_WITH_DYNAMIC_IPP=OFF \
    -D BUILD_FAT_JAVA_LIB=OFF \
    -D BUILD_ANDROID_SERVICE=OFF \
    -D BUILD_CUDA_STUBS=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_OBJC=OFF \
    -D ENABLE_FAST_MATH=ON  \
    -D ENABLE_GNU_STL_DEBUG=ON \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    .. && make -j4 && make install
