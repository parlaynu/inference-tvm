#!/usr/bin/env python3.8
import argparse
import time
import os.path

# from tvm.driver import tvmc
# from tvm.driver.tvmc.model import TVMCModel, TVMCPackage

import inferlib.ops as ops
import inferlib.ops.classify as classify
import inferlib.ops.imaging as imaging
import inferlib.ops.utils as utils
import tvmops.tvm as tvm_ops


def build_pipeline(module, dataspec, rate, limit):
    
    # get the input shape
    shapes, dtypes = module.get_input_info()
    batch_size, nchans, height, width = shapes['image']
    input_dtype = dtypes['image']
    
    print(f"input spec: {shapes['image']} {input_dtype}")

    pipe = ops.datasource(dataspec, resize=(width, height), silent=True)

    if rate > 0:
        pipe = utils.rate_limiter(pipe, rate=rate)
    if limit > 0:
        pipe = utils.limiter(pipe, limit=limit)

    # pipe = imaging.resize(pipe, width=width, height=height)
    pipe = classify.preprocess(pipe)
    pipe = utils.worker(pipe)
    
    pipe = tvm_ops.classify(pipe, module=module)
    pipe = utils.worker(pipe)
    
    pipe = classify.postprocess(pipe)
    
    return pipe


def run(pipe):
    start = time.time()
    
    for idx, item in enumerate(pipe):
        image_id = item['image_id']
        image_size = item['image_size']
        image = item['image']
        
        tops = item['top']
        
        print(f"{idx:02d} {image_id} {image_size} {image.shape}")
        for top, prob in tops:
            print(f"   {top} @ {prob*100.0:0.2f}")
    
    duration = time.time() - start
    
    if item.get('jpeg', None):
        with open("image.jpg", "wb") as f:
            f.write(item['jpeg'])
    
    return duration, idx+1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limit', help='maximum number of images to process', type=int, default=0)
    parser.add_argument('-r', '--rate', help='maximum frame rate for processing', type=int, default=0)
    parser.add_argument('device', help='the device to run on', choices=['cuda', 'cpu'], type=str)
    parser.add_argument('package', help='path to the tvm package or model archive', type=str)
    parser.add_argument('dataspec', help='the data source specification', type=str)
    args = parser.parse_args()
    
    # build it
    module = tvm_ops.prepare_module(args.package, args.device)
    pipe = build_pipeline(module, args.dataspec, args.rate, args.limit)
    
    # and... run it
    duration, count = run(pipe)

    print(f"runtime: {int(duration)} seconds")
    print(f"    fps: {count/duration:0.2f}")


if __name__ == "__main__":
    main()
