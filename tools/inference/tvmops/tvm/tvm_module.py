import tarfile
import tempfile
import os.path

from tvm import rpc
from tvm.contrib import graph_executor as executor


def prepare_module(package_path, device):
    
    with tempfile.TemporaryDirectory() as tempdir:
    
        # check to see if the path is to a model
        with tarfile.open(package_path) as t:
            contents = t.getnames()
            if 'model_package.tar' in contents:
                t.extractall(path=tempdir)
                package_path = os.path.join(tempdir, "model_package.tar")
        

        # here we have a package one way or another
        with tarfile.open(package_path) as t:
            t.extractall(path=tempdir)
            
            lib_file = os.path.join(tempdir, "mod.so")
            graph_file = os.path.join(tempdir, "mod.json")
            params_file = os.path.join(tempdir, "mod.params")
            
            if not os.path.isfile(lib_file):
                raise ValueError("no library in package")
            if not os.path.isfile(graph_file):
                raise ValueError("no graph specification in package")
            if not os.path.isfile(params_file):
                raise ValueError("no parameters in package")

            with open(graph_file) as gf:
                graph = gf.read()
            with open(params_file, "rb") as pf:
                params = bytearray(pf.read())

            session = rpc.LocalSession()
            lib = session.load_module(lib_file)
            
            if device == "cuda":
                dev = session.cuda()
            else:
                assert device == "cpu"
                dev = session.cpu()
            
            module = executor.create(graph, lib, dev)
            module.load_params(params)
                                    
            return module
            

