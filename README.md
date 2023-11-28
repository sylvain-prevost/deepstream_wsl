# deepstream_wsl

## How to visualize graphical rendering on Windows host machine when using Deepstream SDK via WSL

The following is for DeepStream v6.3 running within a WSL2 instance with host having an nVidia dGPU.
(for v6.2, [click here](./README_6_2.md))

https://github.com/sylvain-prevost/deepstream_wsl/assets/10203873/aa01e108-2e94-4b9b-be23-587fdedee723

A small number of minor changes are required.

This is a work-around for lack of EGL symbols in the cuda for WSL2 library - it is extremelly basic as it consist in recompiling without the EGL calls (which can be safely excluded as they relate solely to Jetson devices).

Additionally a small number of changes are required as part of the deepstream pipeline to avoid using gstreamer element whose sources are not available (no simple way to update them).

Example of modifications of gst plugins are provided, as well as example of pipeline modification for 1 of the basic deepstream example (deepstream_test1).

## List of adjusted gst plugins:
- gst-nvinfer
- gst-nvdspreprocess

For each of these plugins the build & install command is:
```bash
$sudo make clean install CUDA_VER=12.1 ENABLE_WSL2=1
```

## List of adjusted samples (C++ & Python):
- deepstream-test1
- deepstream-test2

For each of the C++ examples the build command is:
```bash
$sudo make CUDA_VER=12.1 ENABLE_WSL2=1
```

In order to visualize on host, one need to redirect the display.. there are many ways to do it.. here is just one example:
- Using VcXsrv (https://sourceforge.net/projects/vcxsrv/)
    - launch using 'XLaunch', select default options ()'multiple windows', 'start with no client') + check 'disable access control' (to simplify your life)
- within your WSL instance:
```bash
$export HOST_IP=$(ip route|grep default|awk '{print $3}')
$export DISPLAY=$HOST_IP:0.0
```

Finally, start the test app (C++):
```bash
$sudo ./deepstream-test1-app [path-to-input-stream]
```

#

Per SOFTWARE LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS, 1.2 (iii):
```
This software contains source code provided by NVIDIA Corporation.
```

#

## Repo install/setup detailled process (alternatively you can use 'sudo [install.sh](./install.sh)')

### Repo download and nvidia samples update
``` bash
$cd ~
$git clone https://github.com/sylvain-prevost/deepstream_wsl.git
$cd deepstream_wsl
$sudo cp -r sources_6_3/. /opt/nvidia/deepstream/deepstream-6.3/sources
```

### C++

Adjust nVidia plugins & samples

Compile/link gst-nvinfer plugin
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/gst-plugins/gst-nvinfer
$sudo make clean install CUDA_VER=12.1 ENABLE_WSL2=1
```

Compile/link gst-nvdspreprocess plugin
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/gst-plugins/gst-nvdspreprocess
$sudo make clean install CUDA_VER=12.1 ENABLE_WSL2=1
```

Compile/link deepstream-test1 application
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test1
$sudo make CUDA_VER=12.1 ENABLE_WSL2=1
```

Compile/link deepstream-test2 application
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test2
$sudo make CUDA_VER=12.1 ENABLE_WSL2=1
```

## Python

Add support for compiling bindings, wheel, etc.. 
``` bash
$sudo apt install -y python3-pip python3.8-dev cmake g++ build-essential libtool m4 autoconf automake
$sudo apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev 
$sudo apt install -y libglib2.0-dev libglib2.0-dev-bin libgstreamer1.0-dev libgirepository1.0-dev libcairo2-dev
```

Add Deepstream python support (from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
``` bash
$cd ~
$sudo git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps tmp_deepstream_wsl
$sudo cp -v -r ~/tmp_deepstream_wsl/. /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps
$sudo rm -r ~/tmp_deepstream_wsl
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps
$sudo git submodule update --init
```

Compile Bindings, build and install wheel (Note: this is done here as it must include deepstream_wsl plugin sources updates)
``` bash
$cd bindings
$sudo rm -r build
$sudo mkdir build
$cd build
$sudo cmake ..
$sudo make -j$(nproc)
```

Install updated wheel
``` bash
$pip3 install ./pyds-1.1.8-py3-none*.whl
```

## Executing samples

### C++
Start deepstream-test1 application (wsl)
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test1
$sudo ./deepstream-test1-app /opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.h264
```

Start deepstream-test2 application (wsl)
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test2
$sudo ./deepstream-test2-app /opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.h264
```

### Python
Start python deepstream-test1 application (wsl)
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/apps/deepstream-test1
$python3 deepstream_test_1_wsl.py /opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.h264
```

Start python deepstream-test2 application (wsl)
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/apps/deepstream-test2
$python3 deepstream_test_2_wsl.py /opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.h264
```



https://github.com/sylvain-prevost/deepstream_wsl/assets/10203873/c22a1797-7d13-4492-b5c3-b493ad1ff6f7



## Example on how to prepare your Deepstream_v6.3-WSL instance from scratch 
[click here](./Wsl_6_3_prep.md)

