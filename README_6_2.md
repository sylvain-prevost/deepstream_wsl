# deepstream_wsl

## How to visualize graphical rendering on Windows host machine when using Deepstream SDK via WSL

The following is for DeepStream v6.2 running within a WSL2 instance with host having an nVidia dGPU.
(for v6.3, [click here](./README.md))

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
$sudo make clean install CUDA_VER=11.8 ENABLE_WSL2=1
```

## List of adjusted samples:
- deepstream-test1

For each of the examples the build command is:
```bash
$sudo make CUDA_VER=11.8 ENABLE_WSL2=1
```

In order to visualize on host, one need to redirect the display.. there are many ways to do it.. here is just one example:
- Using VcXsrv (https://sourceforge.net/projects/vcxsrv/)
    - launch using 'XLaunch', select default options ()'multiple windows', 'start with no client') + check 'disable access control' (to simplify your life)
- within your WSL instance:
```bash
$export HOST_IP=$(ip route|grep default|awk '{print $3}')
$export DISPLAY=$HOST_IP:0.0
```

Finally, start the test app:
```bash
$sudo ./deepstream-test1-app [path-to-input-stream]
```

#

Per SOFTWARE LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS, 1.2 (iii):
```
This software contains source code provided by NVIDIA Corporation.
```

#

## repo install/setup process

Adjust nVidia plugins & samples
``` bash
$git clone https://github.com/sylvain-prevost/deepstream_wsl.git
$cd deepstream_wsl
$sudo cp -r sources_6_2/. /opt/nvidia/deepstream/deepstream-6.2/sources
```

Compile/link gst-nvinfer plugin
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.2/sources/gst-plugins/gst-nvinfer
$sudo make clean install CUDA_VER=11.8 ENABLE_WSL2=1
```

Compile/link gst-nvdspreprocess plugin
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.2/sources/gst-plugins/gst-nvdspreprocess
$sudo make clean install CUDA_VER=11.8 ENABLE_WSL2=1
```

Compile/link deepstream-test1 application
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.2/sources/apps/sample_apps/deepstream-test1
$sudo make CUDA_VER=11.8 ENABLE_WSL2=1
```

Start deepstream-test1 application
``` bash
$cd /opt/nvidia/deepstream/deepstream-6.2/sources/apps/sample_apps/deepstream-test1
$sudo ./deepstream-test1-app /opt/nvidia/deepstream/deepstream-6.2/samples/streams/sample_720p.h264
```

## Example on how to prepare your Deepstream_v6.2-WSL instance from scratch 

[click here](./Wsl_6_2_prep.md)

