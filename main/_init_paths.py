import sys
import os
from os import path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        try:
            os.environ["PYTHONPATH"] = path + ":" + os.environ["PYTHONPATH"]
        except KeyError:
            os.environ["PYTHONPATH"] = path

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
