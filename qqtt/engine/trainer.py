from qqtt.data import SimpleData
from qqtt.utils import logger, visualize_pc
from qqtt import SpringMassSystem
import open3d as o3d
import numpy as np
import torch


class InvPhyTrainer:
    def __init__(self, data_path, base_dir, device="cuda:0"):
        self.dataset = SimpleData(data_path, base_dir, device=device, visualize=False)
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
        ) = self._init_start(self.dataset.data[0], device=device)
        self.SpringMassSystem = SpringMassSystem(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=5e-5,
            num_substeps=1000,
            spring_Y=3e4,
            dashpot_damping=100,
            drag_damping=3,
            device=device,
        )

    def _init_start(self, pc, radius=0.1, max_neighbours=20, device="cuda:0"):
        # Connect the springs based on the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
        # Find the nearest neighbours to connect the springs
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        spring_flags = np.zeros((len(points), len(points)))
        springs = []
        rest_lengths = []
        vertices = points  # Use the points as the vertices of the springs
        for i in range(len(vertices)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                points[i], radius, max_neighbours
            )
            import pdb
            pdb.set_trace()
            idx = idx[1:]
            for j in idx:
                if spring_flags[i, j] == 0 and spring_flags[j, i] == 0:
                    spring_flags[i, j] = 1
                    spring_flags[j, i] = 1
                    springs.append([i, j])
                    rest_lengths.append(np.linalg.norm(points[i] - points[j]))
        springs = np.array(springs)
        rest_lengths = np.array(rest_lengths)
        masses = np.ones(len(vertices))
        return (
            torch.tensor(vertices, dtype=torch.float32, device=device),
            torch.tensor(springs, dtype=torch.int32, device=device),
            torch.tensor(rest_lengths, dtype=torch.float32, device=device),
            torch.tensor(masses, dtype=torch.float32, device=device),
        )
    
    def train(self):
        # Train the model with the physical simulator
        pass

    def visualize_sim(self):
        # Visualize the whole simulation using current set of parameters in the physical simulator
        with torch.no_grad():
            # Need to reset the simulator to the initial state
            self.SpringMassSystem.reset_system(
                self.init_vertices,
                self.init_springs,
                self.init_rest_lengths,
                self.init_masses,
            )

            vertices = [self.init_vertices.cpu()]

            frame_len = self.dataset.frame_len
            for i in range(frame_len - 1):
                x, _, _, _ = self.SpringMassSystem.step()
                vertices.append(x.cpu())

            vertices = torch.stack(vertices, dim=0)
            visualize_pc(
                vertices,
                visualize=True,
            )
