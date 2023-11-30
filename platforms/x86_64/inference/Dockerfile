FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN mkdir -p /workspace

WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3.8-venv \
                            llvm-11 libopenblas-dev \
                            git wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

COPY local/packages/tvm-0.14.0-cp38-cp38-linux_x86_64.whl /workspace/
RUN pip3.8 install tvm-0.14.0-cp38-cp38-linux_x86_64.whl && \
    rm -f tvm-0.14.0-cp38-cp38-linux_x86_64.whl

COPY files/requirements.txt requirements.txt
RUN pip3.8 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

RUN git clone https://github.com/parlaynu/inference-onnx.git && \
    mv inference-onnx/tools/inference . && \
    rm -rf inference-onnx

COPY local/inference /workspace/inference/

ENTRYPOINT ["/bin/sh", "-c", "exec \"$0\" \"$@\""]
CMD ["/bin/bash"]

