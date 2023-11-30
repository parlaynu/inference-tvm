# Edge Inference

## General

To get the target string:

    $ llc --version | grep "Host CPU"
      Host CPU: znver2


TensorRT

* https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html


## Resources

###  Python3.8

Build instructions from [here](https://itheo.tech/install-python-38-on-a-raspberry-pi)

If using ARM Compute Library, build with the same compiler as above.

    $ sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev

    $ wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz
    $ tar xzf Python-3.8.18.tgz
    $ cd Python-3.8.18

    $ ./configure --enable-optimizations --prefix=/usr/local/python3.8
    $ make -j4
    $ sudo make install

### Building tvm

Build instructions from [here](https://tvm.apache.org/docs/install/from_source.html)

Use the same compiler as in earlier steps as it needs to match the python3.8 executable so the python tools here
can import the tvm packages.

* https://tvm.apache.org/docs//v0.13.0/how_to/deploy/arm_compute_lib.html


## Systems

Tables of the systems - one table, a row for each system:

* General description
* CPU - name, architecture, llc output, bogomips
* GPU
* RAM
* OS - 32/64 bit
* Versions of software

## AMD Ryzen CPU/GeForce RTX 2060 SUPER

### Resnet18

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet18.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 4 seconds
        fps: 62.31

#### CPU

    $ lscpu | grep "Model name"
    Model name: AMD Ryzen 7 3700X 8-Core Processor

    $ llc --version | grep "Host CPU"
      Host CPU: znver2

    $ ./convert.py ../../models/resnet18.onnx llvm
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 5 seconds
        fps: 48.57

    $ ./convert.py --tune ../../models/resnet18.onnx llvm
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 4 seconds
        fps: 60.05

    $ ./convert.py ../../models/resnet18.onnx "llvm -mcpu=znver2"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 3 seconds
        fps: 74.02

    $ ./convert.py --tune ../../models/resnet18.onnx "llvm -mcpu=znver2"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 2 seconds
        fps: 112.13

#### CUDA

    $ ./convert.py ../../models/resnet18.onnx cuda
    $ ./infer-tvm.py -l 1000 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 2 seconds
        fps: 481.80

    $ ./convert.py --tune ../../models/resnet18.onnx cuda
    $ ./infer-tvm.py -l 1000 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 1 seconds
        fps: 574.17


### Resnet50

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet50.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 8 seconds
        fps: 28.61

#### CPU

    $ lscpu | grep "Model name"
    Model name: AMD Ryzen 7 3700X 8-Core Processor

    $ llc --version | grep "Host CPU"
      Host CPU: znver2

    $ ./convert.py ../../models/resnet50.onnx llvm
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 12 seconds
        fps: 20.18

    $ ./convert.py ../../models/resnet50.onnx "llvm -mcpu=znver2"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 6 seconds
        fps: 36.25

    $ ./convert.py --tune ../../models/resnet50.onnx "llvm -mcpu=znver2"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 5 seconds
        fps: 44.25

#### CUDA

    $ ./convert.py ../../models/resnet50.onnx cuda
    $ ./infer-tvm.py -l 1000 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 5 seconds
        fps: 176.90

    $ ./convert.py --tune ../../models/resnet50.onnx cuda
    $ ./infer-tvm.py -l 1000 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 4 seconds
        fps: 235.57

## Jetson Nano

NOTE: GPU tuning is crashing before it completes

    $ llc-10 --version | grep "Host CPU"
      Host CPU: cortex-a57

### Resnet18

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet18.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 42 seconds
        fps: 5.90

#### CPU

    $ ./convert.py ../../models/resnet18.onnx llvm
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 78 seconds
        fps: 3.19

    $ ./convert.py ../../models/resnet18.onnx "llvm -mcpu=cortex-a57"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 77 seconds
        fps: 3.24

    $ ./convert.py --tune ../../models/resnet18.onnx "llvm -mcpu=cortex-a57"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 34 seconds
        fps: 7.29

#### CUDA

    $ ./convert.py ../../models/resnet18.onnx cuda
    $ ./infer-tvm.py -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 11 seconds
        fps: 22.60

### Resnet50

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet50.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 92 seconds
        fps: 2.70

#### CPU

    $ ./convert.py ../../models/resnet50.onnx "llvm -mcpu=cortex-a57"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 151 seconds
        fps: 1.65

    $ ./convert.py --tune ../../models/resnet50.onnx "llvm -mcpu=cortex-a57"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 82 seconds
        fps: 3.03

#### CUDA

    $ ./convert.py ../../models/resnet50.onnx cuda
    $ ./infer-tvm.py -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 34 seconds
        fps: 7.30

## RaspberryPi 4b - Bookworm - 64bit

    $ llc-10 --version | grep "Host CPU"
      Host CPU: cortex-a72

### Arm Compute Library

Instructions from [here](https://tvm.apache.org/docs//v0.13.0/how_to/deploy/arm_compute_lib.html)

Download the compiled [library](https://github.com/ARM-software/ComputeLibrary/releases/download/v21.08/arm_compute-v21.08-bin-linux-arm64-v8a-neon.tar.gz)


### Resnet18

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet18.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 40 seconds
        fps: 6.22

#### CPU

    $ ./convert.py ../../models/resnet18.onnx llvm
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 59 seconds
        fps: 4.22

    $ ./convert.py ../../models/resnet18.onnx "llvm -mcpu=cortex-a72"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 58 seconds
        fps: 4.26

    $ ./convert.py --tune ../../models/resnet18.onnx "llvm -mcpu=cortex-a72"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 31 seconds
        fps: 8.02

    $ ./convert.py ../../models/resnet18.onnx "llvm -mcpu=cortex-a72 -mattr=+neon"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 56 seconds
        fps: 4.42

    $ ./convert.py --tune ../../models/resnet18.onnx "llvm -mcpu=cortex-a72 -mattr=+neon"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 30 seconds
        fps: 8.11

### Resnet50

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet50.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 99 seconds
        fps: 2.52

#### CPU

    $ ./convert.py ../../models/resnet50.onnx llvm
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 128 seconds
        fps: 1.94

    $ ./convert.py ../../models/resnet50.onnx "llvm -mcpu=cortex-a72"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 127 seconds
        fps: 1.96

    $ ./convert.py --tune ../../models/resnet50.onnx "llvm -mcpu=cortex-a72"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 75 seconds
        fps: 3.32

    $ ./convert.py ../../models/resnet50.onnx "llvm -mcpu=cortex-a72 -mattr=+neon"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 131 seconds
        fps: 1.90

    $ ./convert.py --tune ../../models/resnet50.onnx "llvm -mcpu=cortex-a72 -mattr=+neon"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 76 seconds
        fps: 3.26

## RaspberryPi 4b - Bookworm - 32bit

Checking the cpu:

    $ llc-15 --version | grep "Host CPU"
      Host CPU: cortex-a72

### Building Dependencies

Notes:

* ONNXRuntime needs cmake 3.26 or higher
* python wheels not available on pypi.org for linux_armv7l

System packages

    apt install gfortran protobuf-compiler
    apt install libjpeg-dev zlib1g-dev libtiff-dev libopenblas-dev liblapack-dev pkg-config
    apt install libprotobuf-dev protobuf-compiler

    pip3 install wheel cython==0.29.36 pybind11 pythran scikit-build

Parallel builds:

    export MAKEFLAGS=-j4

Cmake:

    wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5.tar.gz
    cd cmake-3.26.5
    ./configure --prefix=/usr/local/cmake-3.26.5
    make -j4
    sudo make install

Pillow:

    wget https://files.pythonhosted.org/packages/80/d7/c4b258c9098b469c4a4e77b0a99b5f4fd21e359c2e486c977d231f52fc71/Pillow-10.1.0.tar.gz
    tar xzf Pillow-10.1.0.tar.gz
    cd Pillow-10.1.0
    python3.8 setup.py bdist_wheel

Numpy:

    wget https://files.pythonhosted.org/packages/a4/9b/027bec52c633f6556dba6b722d9a0befb40498b9ceddd29cbe67a45a127c/numpy-1.24.4.tar.gz
    tar xzf numpy-1.24.4.tar.gz
    cd numpy-1.24.4
    python3.8 setup.py bdist_wheel

Scipy:

    wget https://files.pythonhosted.org/packages/84/a9/2bf119f3f9cff1f376f924e39cfae18dec92a1514784046d185731301281/scipy-1.10.1.tar.gz
    tar xzf scipy-1.10.1.tar.gz
    cd scipy-1.10.1
    python3.8 setup.py bdist_wheel

OpenCV:

    wget https://files.pythonhosted.org/packages/0c/3a/062caf910026a174b15a24b959bc401e756fda7f42d9d404dec8b166f359/opencv-python-headless-4.8.1.78.tar.gz
    tar xzf opencv-python-headless-4.8.1.78.tar.gz
    cd opencv-python-headless-4.8.1.78
    python3.8 setup.py bdist_wheel

ONNXRuntime:

    NOTE: the cmake_extra_definitions on the command line doesn't seem to get pushed into the 
    dependency pytorch_cpuinfo-src. Worked around it by adding `SET(CMAKE_SYSTEM_PROCESSOR "armv7")` 
    at the top of its `CMakeLists.txt` file.

    git clone https://github.com/microsoft/onnxruntime.git
    cd onnxruntime
    git checkout v1.15.1
    git submodule init
    git submodule update

    MAKEFLAGS=-j2 ./build.sh \
      --config=Release \
      --build_shared_lib \
      --build_wheel \
      --skip_tests \
      --compile_no_warning_as_error \
      --cmake_extra_defines CMAKE_SYSTEM_PROCESSOR=armv7 

    Options to Review:
      --use_acl [{ACL_1902,ACL_1905,ACL_1908,ACL_2002}]
                            Build with ACL for ARM architectures.
      --acl_home ACL_HOME   Path to ACL home dir
      --acl_libs ACL_LIBS   Path to ACL libraries

ONNX:

    instructions: https://github.com/onnx/onnx#linux

    git clone https://github.com/onnx/onnx.git
    cd onnx
    git checkout v1.14.1
    git submodule update --init --recursive
    export CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON
    pip install .

Xgboost

    wget https://files.pythonhosted.org/packages/19/fe/327b4a56ef3e3843b97537ff60381cc4d57a8be7ee99375a8710ee690cb2/xgboost-1.7.6.tar.gz
    tar xzf xgboost-1.7.6.tar.gz
    cd xgboost-1.7.6
    python3.8 setup.py bdist_wheel

### Resnet18

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet18.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 63 seconds
        fps: 3.92

#### CPU

    $ ./convert.py ../../models/resnet18.onnx "llvm -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 20 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 105 seconds
        fps: 0.10

    $ ./convert.py ../../models/resnet18.onnx "llvm -mcpu=cortex-a72 -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 66 seconds
        fps: 3.76

    $ ./convert.py --tune ../../models/resnet18.onnx "llvm -mcpu=cortex-a72 -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 45 seconds
        fps: 5.84

    $ ./convert.py ../../models/resnet18.onnx "llvm -mcpu=cortex-a72 -mattr=+neon -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 62 seconds
        fps: 4.02

    $ ./convert.py --tune ../../models/resnet18.onnx "llvm -mcpu=cortex-a72 -mattr=+neon -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet18-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 44 seconds
        fps: 5.68

### Resnet50

#### ONNX Baseline

    $ ./infer-onnx.py -l 250 ../models/resnet50.onnx ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 141 seconds
        fps: 1.77

#### CPU

    $ ./convert.py ../../models/resnet50.onnx "llvm -mcpu=cortex-a72 -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 154 seconds
        fps: 1.61

    $ ./convert.py --tune ../../models/resnet50.onnx "llvm -mcpu=cortex-a72 -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 102 seconds
        fps: 2.51

    $ ./convert.py ../../models/resnet50.onnx "llvm -mcpu=cortex-a72 -mattr=+neon -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 152 seconds
        fps: 1.64

    $ ./convert.py --tune ../../models/resnet50.onnx "llvm -mcpu=cortex-a72 -mattr=+neon -mfloat-abi=hard"
    $ ./infer-tvm.py -c -l 250 ../models/resnet50-tuned.tar ~/Projects/datasets/kaggle-car-detection/training_images/
    runtime: 110 seconds
        fps: 2.26
