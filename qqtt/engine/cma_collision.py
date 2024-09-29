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


class InvPhyTrainerCMA:
    def __init__(
        self, data_path, base_dir, mask_path=None, velocity_path=None, device="cuda:0"
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        # Load the data
        self.dataset = SimpleData(visualize=False)
        self.init_masks = None
        self.init_velocities = None
        if mask_path is not None:
            mask = np.load(mask_path)
            self.init_masks = torch.tensor(mask, dtype=torch.float32, device=cfg.device)
        if velocity_path is not None:
            velocity = np.load(velocity_path)
            self.init_velocities = torch.tensor(
                velocity, dtype=torch.float32, device=cfg.device
            )
        # Initialize the vertices, springs, rest lengths and masses
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
        ) = self._init_start(
            self.dataset.data[0],
            radius=cfg.radius,
            max_neighbours=cfg.max_neighbours,
            mask=self.init_masks,
        )
        # Initialize the physical simulator
        self.simulator = SpringMassSystem(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.init_collide_elas,
            collide_fric=cfg.init_collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            init_velocities=self.init_velocities,
        )

    def _init_start(self, pc, radius=0.1, max_neighbours=20, mask=None):
        if mask is None:
            # Connect the springs based on the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
            # Find the nearest neighbours to connect the springs
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            points = np.asarray(pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
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
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
            )
        else:
            mask = mask.cpu().numpy()
            points = pc.cpu().numpy()
            # Get the unique value in masks and loop
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            for value in unique_values:
                temp_points = points[mask == value]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(temp_points)
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        temp_points[i], radius, max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        if spring_flags[i, j] == 0 and spring_flags[j, i] == 0:
                            spring_flags[i, j] = 1
                            spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(
                                np.linalg.norm(temp_points[i] - temp_points[j])
                            )
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
            )

    def optimize_collision(self, model_path, max_iter=50):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)
        self.simulator.load_state_dict(checkpoint["model_state_dict"])
        self.simulator.to(cfg.device)

        init_collide_elas = torch.clamp(self.simulator.collide_elas, 0.0, 1.0).item()
        init_collide_fric = torch.clamp(self.simulator.collide_fric, 0.0, 2.0).item()
        init_collide_object_elas = torch.clamp(
            self.simulator.collide_object_elas, 0.0, 1.0
        ).item()
        init_collide_object_fric = torch.clamp(
            self.simulator.collide_object_fric, 0.0, 2.0
        ).item()

        x_init = [
            init_collide_elas,
            init_collide_fric / 2,
            init_collide_object_elas,
            init_collide_object_fric / 2,
        ]
        std = 1 / 6
        es = cma.CMAEvolutionStrategy(x_init, std, {"bounds": [0.0, 1.0], "seed": 42})
        es.optimize(self.error_func, iterations=max_iter)

        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]

        print(f"Optimal x: {optimal_x}, Optimal error: {optimal_error}")
        final_collide_elas = optimal_x[0]
        final_collide_fric = optimal_x[1] * 2
        final_collide_object_elas = optimal_x[2]
        final_collide_object_fric = optimal_x[3] * 2
        print(
            f"Final collide_elas: {final_collide_elas}, final_collide_fric: {final_collide_fric}, final_collide_object_elas: {final_collide_object_elas}, final_collide_object_fric: {final_collide_object_fric}"
        )
        self.simulator.collide_elas.data = torch.tensor(
                final_collide_elas, dtype=torch.float32, device=cfg.device
            )
        self.simulator.collide_fric.data = torch.tensor(
            final_collide_fric, dtype=torch.float32, device=cfg.device
        )
        self.simulator.collide_object_elas.data = torch.tensor(
            final_collide_object_elas, dtype=torch.float32, device=cfg.device
        )
        self.simulator.collide_object_fric.data = torch.tensor(
            final_collide_object_fric, dtype=torch.float32, device=cfg.device
        )
        self.visualize_sim(save_only=True, video_path=os.path.join(cfg.base_dir, "final.mp4"))

    def compute_points_loss(self, gt, x):
        # Compute the mse loss between the ground truth and the predicted points
        return smooth_l1_loss(x, gt, beta=1.0, reduction="mean")

    def error_func(self, collision_parameters):
        with torch.no_grad():
            self.simulator.collide_elas.data = torch.tensor(
                collision_parameters[0], dtype=torch.float32, device=cfg.device
            )
            self.simulator.collide_fric.data = torch.tensor(
                collision_parameters[1] * 2, dtype=torch.float32, device=cfg.device
            )
            self.simulator.collide_object_elas.data = torch.tensor(
                collision_parameters[2], dtype=torch.float32, device=cfg.device
            )
            self.simulator.collide_object_fric.data = torch.tensor(
                collision_parameters[3] * 2, dtype=torch.float32, device=cfg.device
            )

            self.simulator.reset_system(
                self.init_vertices.clone(),
                self.init_springs.clone(),
                self.init_rest_lengths.clone(),
                self.init_masses.clone(),
                initial_velocities=(
                    self.init_velocities.clone()
                    if self.init_velocities is not None
                    else None
                ),
            )
            total_loss = 0.0
            for j in range(1, self.dataset.frame_len):
                x, _, _, _ = self.simulator.step()
                loss = self.compute_points_loss(self.dataset.data[j], x)
                total_loss += loss.item()
            total_loss /= self.dataset.frame_len - 1
            return total_loss


    def visualize_sim(self, save_only=True, video_path=None):
        # Visualize the whole simulation using current set of parameters in the physical simulator
        with torch.no_grad():
            # Need to reset the simulator to the initial state
            self.simulator.reset_system(
                self.init_vertices.clone(),
                self.init_springs.clone(),
                self.init_rest_lengths.clone(),
                self.init_masses.clone(),
                initial_velocities=(
                    self.init_velocities.clone()
                    if self.init_velocities is not None
                    else None
                ),
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
