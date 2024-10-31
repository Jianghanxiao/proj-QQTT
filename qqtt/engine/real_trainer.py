from qqtt.data import RealData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt import SpringMassSystem
import open3d as o3d
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
import wandb
import os
from pytorch3d.loss import chamfer_distance