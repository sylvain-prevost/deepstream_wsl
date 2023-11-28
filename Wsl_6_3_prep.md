## Example on how to prepare your v6.3 Deepstream-WSL instance from scratch  

Obtaining Ubuntu-20.04 base-distro (and save it in another location)  

```bash
>wsl --install -d Ubuntu-20.04
>wsl --export Ubuntu-20.04 [path-to-dest-tar-file (for ex: c:\deepstream_wsl.tar)]
>wsl --unregister Ubuntu-20.04
>wsl --import [your_distro_name] [new-distro-path (for ex: c:\wsl\[your_distro_name])] [path-to-src-tar-file]
```

Launching your wsl distro  
```bash
>wsl -d [distro_name]
```

Setting auto user log & standard work-around VPN setup  
make sure to replace USER_NAME with appropriate value  
```bash
$sudo su USER_NAME
$sudo rm /etc/resolv.conf
$sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
$sudo bash -c 'echo -e "[user]\ndefault = USER_NAME\n[network]\ngenerateResolvConf = false\n" > /etc/wsl.conf'
$sudo chattr +i /etc/resolv.conf
$exit
$exit
```

Stop and Relaunch  
```bash
>wsl --shutdown
>wsl -d [distro_name]
```

## At that point you can: either run script ([Wsl_6.3_prep.sh](./Wsl_6_3_prep.sh))

```bash
$cd ~
$chmod +x Wsl_6_3_prep.sh
$sudo ./Wsl_6_3_prep.sh
```

## Or step-by-step

Handle /usr/lib/wsl/lib/libcuda.so soft-link warning
```bash
$sudo rm /usr/lib/wsl/lib/libcuda.so /usr/lib/wsl/lib/libcuda.so.1
$sudo ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so
$sudo ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so.1
```

Default Upgrade  
```bash
$sudo apt update
$sudo apt -y upgrade
```

Install dependencies required for Deepstream 6.3 (from https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu)  

```bash
$sudo apt install -y libssl1.1
$sudo apt install -y libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0
$sudo apt install -y libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler gcc make git python3
```

Install Cuda toolkit 12.1 for WSL (= does not install linux dGPU driver)
```bash
$wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
$sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
$wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
$sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
$sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
$sudo apt-get update
$sudo apt-get install -y cuda
```

Install TensorRT 8.5.3.1 (note: '+cuda11.8' is not a typo)
```bash
$sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
$sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
$sudo apt-get update
$sudo apt-get install -y libnvinfer8=8.5.3-1+cuda11.8 libnvinfer-plugin8=8.5.3-1+cuda11.8 libnvparsers8=8.5.3-1+cuda11.8 \
libnvonnxparsers8=8.5.3-1+cuda11.8 libnvinfer-bin=8.5.3-1+cuda11.8 libnvinfer-dev=8.5.3-1+cuda11.8 \
libnvinfer-plugin-dev=8.5.3-1+cuda11.8 libnvparsers-dev=8.5.3-1+cuda11.8 libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
libnvinfer-samples=8.5.3-1+cuda11.8 libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 \
python3-libnvinfer=8.5.3-1+cuda11.8 python3-libnvinfer-dev=8.5.3-1+cuda11.8
```

Install Deepstream v6.3
```bash
$sudo wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/deepstream/versions/6.3/files/deepstream-6.3_6.3.0-1_amd64.deb'
$sudo apt-get install ./deepstream-6.3_6.3.0-1_amd64.deb
```

