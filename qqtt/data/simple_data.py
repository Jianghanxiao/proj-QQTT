# The simplest test data with full 3D point trajectories (n_frames, n_points, 3)
import numpy as np
import torch
from qqtt.utils import logger

class SimpleData:
    def __init__(self, data_path, device='cuda'):
        logger.info(f"[DATA]: loading data from {data_path}")

        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=device)
        import pdb
        pdb.set_trace()
        self.frame_len = self.data.shape[0]
        self.point_num = self.data.shape[1]

    def visualize_data(self):
        pass