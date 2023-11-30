#!/usr/bin/env bash

# make sure we're in the right location
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${RUN_DIR}

# a stamp to tag the image with
STAMP=$(date +%s)

# collect the files needed
rm -rf local
mkdir -p local
cp -r ../../../tools/onnx2tvm local

# build the image
docker build \
    --tag local/onnx2tvm:${STAMP} \
    --tag local/onnx2tvm:latest \
    .

# copy out the tvm whl
mkdir -p "${HOME}/Workspace/packages"
docker run -it --rm --network=host \
    -v "${HOME}/Workspace/packages":/workspace/packages \
    local/onnx2tvm:latest \
    cp /workspace/apache-tvm/apache-tvm-src-v0.14.0/python/dist/tvm-0.14.0-cp38-cp38-linux_aarch64.whl /workspace/packages

