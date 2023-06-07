# deepstream_wsl

## How to visualize graphical rendering on Windows host machine when using Deepstream SDK via WSL

A small number of minor changes are required.

This is a work-around for lack of EGL symbols in the cuda for WSL2 library - it is extremelly basic as it consist in recompiling without the EGL calls (which can be safely excluded as they relate solely to Jetson devices).

Additionally a small number of changes are required as part of the deepstream pipeline to avoid using gstreamer element whose sources are not available (no simple way to update them).

Exmaple of modifications of gst plugins are provided, as well as example of pipeline modification for 1 of the basic deepstream example (deepstream_test1).

## List of adjusted gst plugins:
- gst-nvinfer
- gst-nvdspreprocess

For each of these plugins the build & install command is:
```bash
$sudo make clean install CUDA_VER=11.8 ENABLE_WSL2=1
```

## List of adjusted sample:
- deepstream-test1

For each of the examples the build command is:
```bash
$sudo make clean CUDA_VER=11.8 ENABLE_WSL2=1
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
sudo ./deepstream-test1-app [path-to-input-stream]
```



## Example on how to prepare your Deepstream-WSL instance from scratch 

[click here](./Wsl_prep.md)

