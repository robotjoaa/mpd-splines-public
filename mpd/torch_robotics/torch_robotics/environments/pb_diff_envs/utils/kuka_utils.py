import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#import pybullet as p
from typing import Union
import pdb
import os.path as osp

def from_rel2abs_path(abs_fname, rel_path):
    current_dir = osp.dirname(osp.abspath(abs_fname))
    return osp.join(current_dir, rel_path)