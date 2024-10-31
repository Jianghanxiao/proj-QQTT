from qqtt.data import RealData
from qqtt.utils import logger, visualize_pc_real, cfg
from qqtt import SpringMassSystem
import open3d as o3d
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
import wandb
import os
from pytorch3d.loss import chamfer_distance


class RealInvPhyTrainer:
    def __init__(
        self, data_path, base_dir, mask_path=None, velocity_path=None, device="cuda:0"
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        # Load the data
        self.dataset = RealData(visualize=False)
        self.init_masks = None
        self.init_velocities = None

        # Get the object points and controller points
        self.object_points = self.dataset.object_points
        self.object_colors = self.dataset.object_colors
        self.object_visibilities = self.dataset.object_visibilities
        self.object_motions_valid = self.dataset.object_motions_valid
        self.controller_points = self.dataset.controller_points

        import pdb
        pdb.set_trace()