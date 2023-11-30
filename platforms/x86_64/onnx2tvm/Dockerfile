FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN mkdir -p /workspace && \
    mkdir -p /workspace/apache-tvm

WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3.8-dev python3.8-venv \
                            build-essential llvm-11-dev libopenblas-dev \
                            libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
                            git wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh && \
    sh cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr && \
    rm cmake-3.27.9-linux-x86_64.sh

RUN wget https://github.com/apache/tvm/releases/download/v0.14.0/apache-tvm-src-v0.14.0.tar.gz && \
    mv apache-tvm-src-v0.14.0.tar.gz apache-tvm

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

COPY files/requirements.txt requirements.txt
RUN pip3.8 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

COPY files/make-tvm.sh /workspace/apache-tvm
RUN /workspace/apache-tvm/make-tvm.sh

COPY local/onnx2tvm /workspace

ENTRYPOINT ["/bin/sh", "-c", "exec \"$0\" \"$@\""]
CMD ["/bin/bash"]

