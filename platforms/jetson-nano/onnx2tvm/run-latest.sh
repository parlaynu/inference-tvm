#!/usr/bin/env bash

mkdir -p "${HOME}/Workspace/models"
mkdir -p "${HOME}/Workspace/packages"

docker run -it --rm --network=host --runtime=nvidia --gpus all \
    -v "${HOME}/Workspace/models":/workspace/models \
    -v "${HOME}/Workspace/packages":/workspace/packages \
    local/onnx2tvm:latest /bin/bash

