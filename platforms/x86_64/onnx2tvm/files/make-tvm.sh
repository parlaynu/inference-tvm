#!/usr/bin/env bash

cd /workspace/apache-tvm

rm -rf apache-tvm-src-v0.14.0
tar xzf apache-tvm-src-v0.14.0.tar.gz

cd apache-tvm-src-v0.14.0

mkdir -p build
cp cmake/config.cmake build
cd build

sed -i 's/USE_LLVM OFF/USE_LLVM ON/g' config.cmake
sed -i 's/USE_CUDA OFF/USE_CUDA ON/g' config.cmake
sed -i 's/USE_CUDNN OFF/USE_CUDNN ON/g' config.cmake
sed -i 's/USE_CUBLAS OFF/USE_CUBLAS ON/g' config.cmake
sed -i 's/USE_BLAS none/USE_BLAS openblas/g' config.cmake

cmake ..

make -j $(nproc)

cd ../python
python3.8 setup.py bdist_wheel

cd dist
pip3.8 install *whl
