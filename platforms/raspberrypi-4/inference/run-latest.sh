#!/usr/bin/env bash

mkdir -p "${HOME}/Workspace/models"

docker run -it --rm --network=host \
    -v "${HOME}/Workspace/models":/workspace/models \
    local/inference-tvm:latest /bin/bash


