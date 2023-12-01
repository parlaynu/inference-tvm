FROM nvcr.io/nvidia/l4t-cuda:10.2.460-runtime

RUN mkdir -p /workspace

WORKDIR /workspace

COPY local/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/
COPY local/jetson-ota-public.asc /etc/apt/trusted.gpg.d/
RUN rm -f /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3.8-dev python3.8-venv \
                            build-essential llvm-10 libopenblas-dev \
                            cuda-cudart-10-2 libcudnn8 \
                            git wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

COPY local/packages/tvm-0.14.0-cp38-cp38-linux_aarch64.whl /workspace/
RUN pip3.8 install tvm-0.14.0-cp38-cp38-linux_aarch64.whl && \
    rm -f tvm-0.14.0-cp38-cp38-linux_aarch64.whl

COPY files/requirements.txt requirements.txt
RUN pip3.8 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

RUN git clone https://github.com/parlaynu/inference-onnx.git && \
    mv inference-onnx/tools/inference . && \
    rm -rf inference-onnx

COPY local/inference /workspace/inference/

RUN sed -i 's/env python3/env python3.8/g' /workspace/inference/classify-onnx.py


ENTRYPOINT ["/bin/sh", "-c", "exec \"$0\" \"$@\""]
CMD ["/bin/bash"]

