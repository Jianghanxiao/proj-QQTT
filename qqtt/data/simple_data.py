# The simplest test data with full 3D point trajectories (n_frames, n_points, 3)
import numpy as np
import torch
from qqtt.utils import logger, visualize_pc


class SimpleData:
    def __init__(self, data_path, base_dir, device="cuda:0", visualize=False):
        logger.info(f"[DATA]: loading data from {data_path}")

        self.data_path = data_path
        self.base_dir = base_dir
        self.data = np.load(self.data_path)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=device)
        self.frame_len = self.data.shape[0]
        self.point_num = self.data.shape[1]
        if visualize:
            # Visualize the GT frames
            self.visualize_data()

    def visualize_data(self):
        visualize_pc(
            self.data,
            FPS=10,
            visualize=True,
        )
        visualize_pc(
            self.data,
            FPS=10,
            visualize=False,
            save_video=True,
            save_path=f"{self.base_dir}/gt.mp4",
        )
