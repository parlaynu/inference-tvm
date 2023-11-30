FROM nvcr.io/nvidia/l4t-cuda:10.2.460-runtime

RUN mkdir -p /workspace && \
    mkdir -p /workspace/apache-tvm

WORKDIR /workspace

COPY local/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/
COPY local/jetson-ota-public.asc /etc/apt/trusted.gpg.d/
RUN rm -f /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3.8-dev python3.8-venv \
                            build-essential llvm-10-dev libopenblas-dev \
                            cuda-cudart-dev-10-2 libcudnn8-dev \
                            libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
                            git wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-aarch64.sh && \
    sh cmake-3.27.9-linux-aarch64.sh --skip-license --prefix=/usr && \
    rm cmake-3.27.9-linux-aarch64.sh

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

