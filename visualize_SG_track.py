import glob
import os
import open3d as o3d
import numpy as np
import json
import csv
import pickle
from scipy.spatial import KDTree
from pytorch3d.loss import chamfer_distance