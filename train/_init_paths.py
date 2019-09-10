import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.abspath(osp.dirname(__file__))

# Add Framework to PYTHONPATH
lib_path = osp.join(this_dir, '..')
add_path(lib_path)
