from qqtt.data import SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt import SpringMassSystem
import open3d as o3d
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
import wandb
import os
import cma



class InvPhyTrainerCam:
    def __init__(self, data_path, base_dir, device="cuda:0"):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        # Load the data
        self.dataset = SimpleData(visualize=False)
        # Initialize the vertices, springs, rest lengths and masses
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
        ) = self._init_start(self.dataset.data[0], device=device)
        # Initialize the physical simulator
        self.simulator = SpringMassSystem(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
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
        x_init = [30000, 30000]
        std = 10000
        es = cma.CMAEvolutionStrategy(x_init, std, {'bounds': [0.0, 60000.0]})
        es.optimize(self.error_func, iterations=50)
        # while not es.stop():
        #     solutions = es.ask()
        #     print(solutions, "!!!!!!!")
        #     es.tell(solutions, [self.error_func(x) for x in solutions])
        #     es.logger.add()  # write data to disc to be plotted
        #     es.disp()
        # es.result_pretty()
        # cma.plot()
        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]
        print(f"Optimal x: {optimal_x}, Optimal error: {optimal_error}")

    def error_func(self, current_Y):
        with torch.no_grad():
            self.simulator.reset_system(
                    self.init_vertices,
                    self.init_springs,
                    self.init_rest_lengths,
                    self.init_masses,
                )
            # Compute the error based on the current set of parameters
            self.simulator.spring_Y.data = torch.log(torch.tensor(current_Y[0], dtype=torch.float32))
            total_loss = 0.0
            for j in range(1, self.dataset.frame_len):
                x, _, _, _ = self.simulator.step()
                loss = self.compute_points_loss(self.dataset.data[j], x)
                # self.simulator.detach()
                total_loss += loss.item()
            total_loss /= self.dataset.frame_len - 1
            return total_loss

    def compute_points_loss(self, gt, x):
        # Compute the mse loss between the ground truth and the predicted points
        return smooth_l1_loss(x, gt, beta=1.0, reduction="mean")

    def visualize_sim(self, save_only=True, video_path=None):
        # Visualize the whole simulation using current set of parameters in the physical simulator
        with torch.no_grad():
            # Need to reset the simulator to the initial state
            self.simulator.reset_system(
                self.init_vertices,
                self.init_springs,
                self.init_rest_lengths,
                self.init_masses,
            )

            vertices = [self.init_vertices.cpu()]

            frame_len = self.dataset.frame_len
            for i in range(frame_len - 1):
                x, _, _, _ = self.simulator.step()
                vertices.append(x.cpu())

            vertices = torch.stack(vertices, dim=0)
            if not save_only:
                visualize_pc(
                    vertices,
                    visualize=True,
                )
            else:
                assert video_path is not None, "Please provide the video path to save"
                visualize_pc(
                    vertices,
                    visualize=False,
                    save_video=True,
                    save_path=video_path,
                )

    def visualize_gt(self):
        # Visualize the ground truth data
        self.dataset.visualize_data()
