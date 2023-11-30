# Inference Using ApacheTVM

This repository has tools and guidelines for converting ONNX models to [ApacheTVM](https://tvm.apache.org/)
and running classification inference using the exported model. 

The tools include:

* convert ONNX to ApacheTVM format
* running inference using the exported TVM package

The `tools` directory contains the source code in python for the onnx2tvm conversion and the inference. It builds 
on the tools in [inference-onnx](https://github.com/parlaynu/inference-onnx). Models converted to ONNX using the 
`inference-onnx` project can be used as input to the tools here.

The `platforms` directory contains the tooling to build docker images with the tools and packages to
run the conversion and inference.

Each platform needs to do its own conversion as the ApacheTVM is a binary format with the compiled model
in a loadable library for that platform.

## The Tools

### Convert ONNX to TVM

This tools converts an ONNX model to TVM. 

The full usage is:

    $ ./onnx2tvm.py  -h
    usage: onnx2tvm.py [-h] [-t] model [target]
    
    positional arguments:
      model       model architecture
      target      the target to compile and tune for
      
    optional arguments:
      -h, --help  show this help message and exit
      -t, --tune  tune the model

The containers built by the platform tools mount a directory called `models` from the host file system which
can be used as the source for ONNX model files.

To export for a given CPU, first determine the CPU model:

    $ llc --version | grep "Host CPU"
      Host CPU: znver2

Then run an export for the CPU:

    $ ./onnx2tvm.py models/resnet18-1x3x224x224.onnx "llvm -mcpu=znver2"
    inputs: 1
      00: image [1, 3, 224, 224] tensor(float)
    outputs: 1
      00: preds [1, 1000] tensor(float)
    loading the model...
    compiling the model...
    One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
    saving the model to models/resnet18-1x3x224x224-llvm.tar...

To tune the model use the '-t' flag ... and wait.

To export for CUDA, run the command like this:

    ./onnx2tvm.py models/resnet18-1x3x224x224.onnx cuda               
    inputs: 1
      00: image [1, 3, 224, 224] tensor(float)
    outputs: 1
      00: preds [1, 1000] tensor(float)
    loading the model...
    compiling the model...
    One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
    saving the model to models/resnet18-1x3x224x224-cuda.tar...

Again, to build a tuned model for CUDA, use the `-t` flag ... and wait. I haven't been able to successfully tune a model for the
GPU on the JetsonNano platform - it has always crashed before completing.

### Running Inference

The tool `classify-tvm.py` runs inference on the exported TVM model. The full usage is:

    $ ./classify-tvm.py -h
    usage: classify-tvm.py [-h] [-l LIMIT] [-r RATE] {cuda,cpu} package dataspec
    
    positional arguments:
      {cuda,cpu}            the device to run on
      package               path to the tvm package or model archive
      dataspec              the data source specification
      
    optional arguments:
      -h, --help            show this help message and exit
      -l LIMIT, --limit LIMIT
                            maximum number of images to process
      -r RATE, --rate RATE  maximum frame rate for processing


A simple run using CUDA and a camera server from the `inference-onnx` project looks like this:

    ./classify-tvm.py -l 10 cuda ../models/resnet18-1x3x224x224-cuda.tar tcp://192.168.24.31:8089
    2023-11-30 04:58:45.962 INFO load_module /tmp/tmpv_7ssmml/mod.so
    input spec: [1, 3, 224, 224] float32
    00 image_0000 1920x1080x3 (1, 3, 224, 224)
       315 @ 13.46
    01 image_0001 1920x1080x3 (1, 3, 224, 224)
       315 @ 16.66
    02 image_0002 1920x1080x3 (1, 3, 224, 224)
       315 @ 15.98
    03 image_0003 1920x1080x3 (1, 3, 224, 224)
       315 @ 13.94
    04 image_0004 1920x1080x3 (1, 3, 224, 224)
       315 @ 20.04
    05 image_0005 1920x1080x3 (1, 3, 224, 224)
       315 @ 18.55
    06 image_0006 1920x1080x3 (1, 3, 224, 224)
       315 @ 15.89
    07 image_0007 1920x1080x3 (1, 3, 224, 224)
       315 @ 25.31
    08 image_0008 1920x1080x3 (1, 3, 224, 224)
       315 @ 18.67
    09 image_0009 1920x1080x3 (1, 3, 224, 224)
       315 @ 17.82
    runtime: 1 seconds
        fps: 6.33

See the [inference-onnx](https://github.com/parlaynu/inference-onnx) project for details on the camera server.

## The Platforms

Under the `platforms` directory, there is a directory for each platform supported, and inside each platform,
there is a directory for each tool for that platform.

The README.md files have platform specific setup notes 

### The Conversion Container

In the `onnx2tvm` directory are the tools to build the conversion container and launch it.

Use the `build.sh` script to build the container. This does everything automatically including downloading the 
TVM source code and compiling it and building and installing the python package. This takes some time on the
JetsonNano and RaspberryPi4 platforms.

    ./build.sh

Use the `run-latest.sh` script to launch the container with the correct parameters:

    $ ./run-latest.sh
    
    root@eximius:/workspace# ls
    README.md  apache-tvm  models  onnx2tvm.py

The `models` diretory is mounted from the host system from ${HOME}/Workspace/models. Place any models you want to convert
into this directory so they are accessible from this container.

### The Inference Container

In the `inference` directory at the tools to build the inference container and launch it.

Use the `build.sh` script to build the container. This does everything automatically to create an environment
to run inference.

    ./build.sh

Use the `run-latest.sh` script to launch the container with the correct parameters:

    $ ./run-latest.sh
    
    root@eximius:/workspace# ls
    inference  models
    
    root@eximius:/workspace# ls inference/
    classify-onnx.py  classify-tvm.py  image.jpg  inferlib  tvmops
    
    root@eximius:/workspace# ls models/
    resnet18-1x3x224x224-cuda.tar  resnet18-1x3x224x224-llvm.tar  resnet18-1x3x224x224.onnx  resnet50-1x3x224x224.onnx

The `models` diretory is mounted from the host system from ${HOME}/models. Place any models you want to convert
into this directory so they are accessible from this container.
