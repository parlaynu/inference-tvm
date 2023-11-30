#!/usr/bin/env python3.8
import argparse
import os, os.path
import onnxruntime as ort
import onnx
from tvm.driver import tvmc


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tune", help="tune the model", action="store_true")
    parser.add_argument("model", help="model architecture", type=str)
    parser.add_argument("target", help="the target to compile and tune for", type=str, nargs="?", default="cuda")
    args = parser.parse_args()
    
    return args


def inspect_onnx(onnx_path):
    session = ort.InferenceSession(onnx_path)
    
    return session.get_inputs(), session.get_outputs()

    
def main():
    args = parse_cmdline()
    
    # create the output dir
    outdir = os.path.dirname(args.model)
    if len(outdir) == 0:
        outdir = "./local/models"
    args.outdir = outdir

    os.makedirs(args.outdir, exist_ok=True)
    
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    model_suffix = args.target.split(' ')[0]
    if args.tune:
        save_path = os.path.join(args.outdir, f"{model_name}-{model_suffix}-tuned.tar")
    else:
        save_path = os.path.join(args.outdir, f"{model_name}-{model_suffix}.tar")

    # get info about the model inputs and outputs
    inputs, outputs = inspect_onnx(args.model)
    
    print(f"inputs: {len(inputs)}")
    for idx, i in enumerate(inputs):
        print(f"  {idx:02d}: {i.name} {i.shape} {i.type}")
    print(f"outputs: {len(outputs)}")
    for idx, o in enumerate(outputs):
        print(f"  {idx:02d}: {o.name} {o.shape} {o.type}")

    # start the conversion process
    print("loading the model...", flush=True)
    input_shapes = {
        i.name : i.shape for i in inputs
    }
    model = tvmc.load(args.model, shape_dict=input_shapes)
    
    tuning_records = None
    if args.tune:
        print("tuning the model...", flush=True)
        tuning_records = tvmc.tune(model, target=args.target)
        
    print("compiling the model...", flush=True)
    tvmc.compile(model, target=args.target, tuning_records=tuning_records)
    
    print(f"saving the model to {save_path}...", flush=True)
    model.save(save_path)


if __name__ == "__main__" :
    main()

