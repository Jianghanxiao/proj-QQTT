import numpy as np
import torch
import pickle
from qqtt.utils import logger, visualize_pc_real, cfg


class RealData:
    def __init__(self, visualize=False):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")
        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        controller_points = data["controller_points"]

        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=cfg.device
        )
        self.object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=cfg.device
        )
        # object_visibilities is a binary mask
        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=cfg.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=cfg.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        self.frame_len = self.object_points.shape[0]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize)

    def visualize_data(self, visualize=False):
        if visualize:
            visualize_pc_real(
                self.object_points,
                self.object_colors,
                self.object_visibilities,
                self.object_motions_valid,
                self.controller_points,
                visualize=True,
            )
        visualize_pc_real(
            self.object_points,
            self.object_colors,
            self.object_visibilities,
            self.object_motions_valid,
            self.controller_points,
            visualize=False,
            save_video=True,
            save_path=f"{self.base_dir}/gt.mp4",
        )
