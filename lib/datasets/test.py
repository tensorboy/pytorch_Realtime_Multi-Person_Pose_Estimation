from lib.datasets.datasets import SoybeanKeypoints
import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(ROOT_DIR, 'data/bean')

s = SoybeanKeypoints(data_dir)


