#!/bin/bash

# Handle wsl libcuda soft-link warning
rm /usr/lib/wsl/lib/libcuda.so /usr/lib/wsl/lib/libcuda.so.1
ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so
ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so.1

apt update
apt -y upgrade

# Install dependencies required for Deepstream 6.3 (from https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu)  
apt install -y libssl1.1
apt install -y libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0
apt install -y libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler gcc make git python3

# Install Cuda toolkit 12.1 for WSL (= does not install linux dGPU driver)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get install -y cuda

# Install TensorRT 8.5.3.1 (note: '+cuda11.8' is not a typo)
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update
apt-get install -y libnvinfer8=8.5.3-1+cuda11.8 libnvinfer-plugin8=8.5.3-1+cuda11.8 libnvparsers8=8.5.3-1+cuda11.8 \
libnvonnxparsers8=8.5.3-1+cuda11.8 libnvinfer-bin=8.5.3-1+cuda11.8 libnvinfer-dev=8.5.3-1+cuda11.8 \
libnvinfer-plugin-dev=8.5.3-1+cuda11.8 libnvparsers-dev=8.5.3-1+cuda11.8 libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
libnvinfer-samples=8.5.3-1+cuda11.8 libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 \
python3-libnvinfer=8.5.3-1+cuda11.8 python3-libnvinfer-dev=8.5.3-1+cuda11.8

# Install Deepstream v6.3
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/deepstream/versions/6.3/files/deepstream-6.3_6.3.0-1_amd64.deb'
apt-get install ./deepstream-6.3_6.3.0-1_amd64.deb


