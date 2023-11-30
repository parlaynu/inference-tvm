#!/usr/bin/env bash

mkdir -p "${HOME}/Workspace/models"

docker run -it --rm --network=host \
    -v "${HOME}/Workspace/models":/workspace/models \
    local/onnx2tvm:latest /bin/bash

