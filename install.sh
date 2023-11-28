#!/bin/bash

# obtain wsl-adjusted deepstream 
cd ~
git clone https://github.com/sylvain-prevost/deepstream_wsl.git
cd deepstream_wsl
cp -r sources_6_3/. /opt/nvidia/deepstream/deepstream-6.3/sources

# build nvinfer plugin
cd /opt/nvidia/deepstream/deepstream-6.3/sources/gst-plugins/gst-nvinfer
make clean install CUDA_VER=12.1 ENABLE_WSL2=1

# build nvdspreprocess plugin
cd /opt/nvidia/deepstream/deepstream-6.3/sources/gst-plugins/gst-nvdspreprocess
make clean install CUDA_VER=12.1 ENABLE_WSL2=1

# build deepstream-test1 C++ app
cd /opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test1
make CUDA_VER=12.1 ENABLE_WSL2=1

# build deepstream-test2 C++ app
cd /opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test2
make CUDA_VER=12.1 ENABLE_WSL2=1

# add support for python build
apt install -y python3-pip python3.8-dev cmake g++ build-essential  libtool m4 autoconf automake
apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev 
apt install -y libglib2.0-dev libglib2.0-dev-bin libgstreamer1.0-dev libgirepository1.0-dev libcairo2-dev

# obtain deepstream-python 
cd ~
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps tmp_deepstream_wsl
cp -v -r ~/tmp_deepstream_wsl/. /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps
rm -r ~/tmp_deepstream_wsl
cd /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps
git submodule update --init

# compile bindings
cd /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/bindings
rm -r build
mkdir build
cd build
cmake ..
make -j$(nproc)

# install wsl-adjusted python wheel
pip3 install ./pyds-1.1.8-py3-none*.whl