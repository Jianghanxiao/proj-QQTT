from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp
from scipy.spatial import KDTree
import pickle
import cv2
from pynput import keyboard

# TODO: 3dgs library
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.dynamic_utils import interpolate_motions_feng, interpolate_motions_feng_speedup, knn_weights, knn_weights_sparse, get_topk_indices, calc_weights_vals_from_indices
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gaussian_splatting.render import remove_gaussians_with_low_opacity

from sklearn.cluster import KMeans


class InvPhyTrainerWarp:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        if not pure_inference_mode:
            self.optimizer = torch.optim.Adam(
                [
                    wp.to_torch(self.simulator.wp_spring_Y),
                    wp.to_torch(self.simulator.wp_collide_elas),
                    wp.to_torch(self.simulator.wp_collide_fric),
                    wp.to_torch(self.simulator.wp_collide_object_elas),
                    wp.to_torch(self.simulator.wp_collide_object_fric),
                ],
                lr=cfg.base_lr,
                betas=(0.9, 0.99),
            )

            if "debug" not in cfg.run_name:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="final_pipeline",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Debug",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            if not os.path.exists(f"{cfg.base_dir}/train"):
                # Create directory if it doesn't exist
                os.makedirs(f"{cfg.base_dir}/train")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def train(self, start_epoch=-1):
        # Render the initial visualization
        video_path = f"{cfg.base_dir}/train/init.mp4"
        self.visualize_sim(save_only=True, video_path=video_path)

        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
            total_loss = 0.0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0
                # total_acc_loss = 0.0
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            # if cfg.data_type == "real":
            #     self.simulator.set_acc_count(False)
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, cfg.train_frame)):
                    self.simulator.set_controller_target(j)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        if cfg.data_type == "real":
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_loss()
                            self.simulator.tape.backward(self.simulator.loss)
                        else:
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_simple_loss()
                            self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()

                    if cfg.data_type == "real":
                        chamfer_loss = wp.to_torch(
                            self.simulator.chamfer_loss, requires_grad=False
                        )
                        track_loss = wp.to_torch(
                            self.simulator.track_loss, requires_grad=False
                        )
                        total_chamfer_loss += chamfer_loss.item()
                        total_track_loss += track_loss.item()

                        # if (
                        #     wp.to_torch(self.simulator.acc_count, requires_grad=False)[
                        #         0
                        #     ]
                        #     == 1
                        # ):
                        #     acc_loss = wp.to_torch(
                        #         self.simulator.acc_loss, requires_grad=False
                        #     )
                        #     total_acc_loss += acc_loss.item()
                        # else:
                        #     self.simulator.set_acc_count(True)

                        # # Update the prev_acc used to calculate the acceleration loss
                        # self.simulator.update_acc()

                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()

                    if cfg.use_graph:
                        # Only need to clear the gradient, the tape is created in the graph
                        self.simulator.tape.zero()
                    else:
                        # Need to reset the compute graph and clear the gradient
                        self.simulator.tape.reset()
                    self.simulator.clear_loss()
                    # Set the intial state for the next step
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )

            total_loss /= cfg.train_frame - 1
            if cfg.data_type == "real":
                total_chamfer_loss /= cfg.train_frame - 1
                total_track_loss /= cfg.train_frame - 1
                # total_acc_loss /= cfg.train_frame - 2
            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": (
                        total_chamfer_loss if cfg.data_type == "real" else 0
                    ),
                    "track_loss": total_track_loss if cfg.data_type == "real" else 0,
                    # "acc_loss": total_acc_loss if cfg.data_type == "real" else 0,
                    "collide_else": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ).item(),
                },
                step=i,
            )

            logger.info(f"[Train]: Iteration: {i}, Loss: {total_loss}")

            if i % cfg.vis_interval == 0 or i == cfg.iterations - 1:
                video_path = f"{cfg.base_dir}/train/sim_iter{i}.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
                wandb.log(
                    {
                        "video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=cfg.FPS,
                        ),
                    },
                    step=i,
                )
                # Save the parameters
                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
                    ),
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                if best_loss == None or total_loss < best_loss:
                    # Remove old best model file if it exists
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)

                    # Update best loss and best epoch
                    best_loss = total_loss
                    best_epoch = i

                    # Save new best model
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"Latest best model saved: epoch {best_epoch} with loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")
                logger.info(
                    f"[Visualize]: Visualize the simulation at iteration {i} and save the model"
                )

        wandb.finish()

    def test(self, model_path=None):
        if model_path is not None:
            # Load the model
            logger.info(f"Load model from {model_path}")
            checkpoint = torch.load(model_path, map_location=cfg.device)

            spring_Y = checkpoint["spring_Y"]
            collide_elas = checkpoint["collide_elas"]
            collide_fric = checkpoint["collide_fric"]
            collide_object_elas = checkpoint["collide_object_elas"]
            collide_object_fric = checkpoint["collide_object_fric"]
            num_object_springs = checkpoint["num_object_springs"]

            assert (
                len(spring_Y) == self.simulator.n_springs
            ), "Check if the loaded checkpoint match the config file to connect the springs"

            self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
            self.simulator.set_collide(
                collide_elas.detach().clone(), collide_fric.detach().clone()
            )
            self.simulator.set_collide_object(
                collide_object_elas.detach().clone(),
                collide_object_fric.detach().clone(),
            )

        # Render the initial visualization
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def outdomain_inference(self, model_path, to_case_data_path, final_points):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        spring_Y = spring_Y[: self.num_object_springs]

        # Loda the to_case data
        with open(to_case_data_path, "rb") as f:
            data = pickle.load(f)
        self.object_points = torch.tensor(
            data["object_points"], dtype=torch.float32, device=cfg.device
        )
        self.object_colors = self.dataset.object_colors[0].repeat(
            self.object_points.shape[0], 1, 1
        )

        self.controller_points = torch.tensor(
            data["controller_points"], dtype=torch.float32, device=cfg.device
        )
        self.dataset.frame_len = self.object_points.shape[0]

        controller_points = self.controller_points.cpu().numpy()

        assert self.num_all_points == len(
            final_points
        ), "Check the length of the final points"

        # Update the rest lengths for the springs among the object points
        springs = self.init_springs.cpu().numpy()[: self.num_object_springs]
        rest_lengths = np.linalg.norm(
            final_points[springs[:, 0]] - final_points[springs[:, 1]], axis=1
        )

        springs = springs.tolist()
        rest_lengths = rest_lengths.tolist()
        # Update the connection between the final points and the controller points
        first_frame_controller_points = controller_points[0]
        points = np.concatenate([final_points, first_frame_controller_points], axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Process to get the connection distance among the controller points and object points
        # Locate the nearest object point for each controller point
        kdtree = KDTree(final_points)
        _, idx = kdtree.query(first_frame_controller_points, k=1)
        # find the distances
        distances = np.linalg.norm(
            final_points[idx] - first_frame_controller_points, axis=1
        )
        # find the indices of the top 4 controller points that are close
        top_k = 10
        top_k_idx = np.argsort(distances)[:top_k]
        controller_radius = np.ones(first_frame_controller_points.shape[0]) * 0.01
        controller_radius[top_k_idx] = distances[top_k_idx] + 0.005

        for i in range(len(first_frame_controller_points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                first_frame_controller_points[i],
                controller_radius[i],
                30,
            )
            for j in idx:
                springs.append([self.num_all_points + i, j])
                rest_lengths.append(
                    np.linalg.norm(first_frame_controller_points[i] - points[j])
                )

        self.init_springs = torch.tensor(
            np.array(springs), dtype=torch.int32, device=cfg.device
        )

        self.init_rest_lengths = torch.tensor(
            np.array(rest_lengths), dtype=torch.float32, device=cfg.device
        )
        self.init_masses = torch.tensor(
            np.ones(len(points)), dtype=torch.float32, device=cfg.device
        )

        self.init_vertices = torch.tensor(
            points,
            dtype=torch.float32,
            device=cfg.device,
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        cfg.dt = 5e-6
        cfg.num_substeps = round(1.0 / cfg.FPS / cfg.dt)
        cfg.collision_dist = 0.005

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        spring_Y = torch.cat(
            [
                spring_Y,
                3e4
                * torch.ones(
                    self.simulator.n_springs - self.num_object_springs,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
            ]
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(), collide_object_fric.detach().clone()
        )

        # Render the final results
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def visualize_sim(
        self, save_only=True, video_path=None, save_trajectory=False, save_path=None
    ):
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i, pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_target_change(self):
        target_change = np.zeros((self.n_ctrl_parts, 3))
        for key in self.pressed_keys:
            if key in self.key_mappings:
                idx, change = self.key_mappings[key]
                target_change[idx] += change
        return target_change
    
    def init_control_visualizer(self):

        height = cfg.WH[1]
        width = cfg.WH[0]

        self.arrow_fill = cv2.imread("./assets/arrow_fill.png", cv2.IMREAD_UNCHANGED)
        self.arrow_empty = cv2.imread("./assets/arrow_empty.png", cv2.IMREAD_UNCHANGED)
        
        self.arrow_size = 50
        self.arrow_fill = cv2.resize(self.arrow_fill, (self.arrow_size, self.arrow_size))
        self.arrow_empty = cv2.resize(self.arrow_empty, (self.arrow_size, self.arrow_size))
    
        self.bottom_margin = 50  # Margin from bottom of screen
        bottom_y = height - self.bottom_margin
        top_y = height - (self.bottom_margin + self.arrow_size)
        
        spacing = self.arrow_size

        self.edge_buffer = max(self.arrow_size // 2, 50)
        set1_margin_x = self.edge_buffer                       # Add buffer from left edge
        set2_margin_x = width - self.edge_buffer
        
        self.arrow_positions_set1 = {
            "q": (set1_margin_x + spacing*3, top_y),    # Up
            "w": (set1_margin_x + spacing, top_y),      # Forward
            "a": (set1_margin_x, bottom_y),             # Left
            "s": (set1_margin_x + spacing, bottom_y),   # Backward
            "d": (set1_margin_x + spacing*2, bottom_y), # Right
            "e": (set1_margin_x + spacing*3, bottom_y), # Down
        }
        
        self.arrow_positions_set2 = {
            "u": (set2_margin_x, top_y),                # Up
            "i": (set2_margin_x - spacing*2, top_y),    # Forward
            "j": (set2_margin_x - spacing*3, bottom_y), # Left
            "k": (set2_margin_x - spacing*2, bottom_y), # Backward
            "l": (set2_margin_x - spacing, bottom_y),   # Right
            "o": (set2_margin_x, bottom_y),             # Down
        }
        
        # Create rotation matrices for each arrow
        self.rotations = {
            "w": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 0, 1),   # Forward
            "a": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 90, 1),  # Left
            "s": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 180, 1), # Backward
            "d": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 270, 1), # Right
            "q": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 0, 1),   # Up
            "e": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 180, 1), # Down
            "i": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 0, 1),   # Forward
            "j": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 90, 1),  # Left
            "k": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 180, 1), # Backward
            "l": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 270, 1), # Right
            "u": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 0, 1),   # Up
            "o": cv2.getRotationMatrix2D((self.arrow_size // 2, self.arrow_size // 2), 180, 1), # Down
        }

    def _rotate_arrow(self, arrow, key):
        rotation_matrix = self.rotations[key]
        rotated = cv2.warpAffine(
            arrow,
            rotation_matrix,
            (self.arrow_size, self.arrow_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        return rotated

    def _overlay_arrow(self, background, arrow, position, key):
        x, y = position
        rotated_arrow = self._rotate_arrow(arrow, key)
        
        h, w = rotated_arrow.shape[:2]
        
        roi_x = max(0, x - w // 2)
        roi_y = max(0, y - h // 2)
        roi_w = min(w, background.shape[1] - roi_x)
        roi_h = min(h, background.shape[0] - roi_y)
        
        arrow_x = max(0, w // 2 - x)
        arrow_y = max(0, h // 2 - y)
        
        roi = background[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        arrow_roi = rotated_arrow[arrow_y:arrow_y+roi_h, arrow_x:arrow_x+roi_w]
        
        alpha = arrow_roi[:, :, 3] / 255.0
        
        for c in range(3):  # Apply for RGB channels
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + arrow_roi[:, :, c] * alpha
        
        background[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi
        
        return background
    
    def update_frame(self, frame, pressed_keys):
        result = frame.copy()
        
        # Draw all buttons for Set 1 (left side)
        for key, pos in self.arrow_positions_set1.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result, self.arrow_fill, pos, key)
            else:
                result = self._overlay_arrow(result, self.arrow_empty, pos, key)
        
        # Draw all buttons for Set 2 (right side)
        for key, pos in self.arrow_positions_set2.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result, self.arrow_fill, pos, key)
            else:
                result = self._overlay_arrow(result, self.arrow_empty, pos, key)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size_control1 = cv2.getTextSize("Control 1", font, font_scale, thickness)[0]
        text_size_control2 = cv2.getTextSize("Control 2", font, font_scale, thickness)[0]
        control1_x = self.edge_buffer
        control2_x = cfg.WH[0] - self.edge_buffer - text_size_control2[0]
        text_y = cfg.WH[1] - self.arrow_size*2 - self.bottom_margin
        cv2.putText(result, "Control 1", (control1_x, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(result, "Control 2", (control2_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return result

    def interactive_playground(self, model_path, gs_path, n_ctrl_parts=1):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        ###########################################################################

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)

        vis_vertices = (
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
            .cpu()
            .numpy()
        )

        current_target = self.simulator.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        # TODO: load gaussians from the gs_path
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True        # set to True for white background
        bg_color = [1,1,1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, cfg.WH[1])
        FovX = focal2fov(focal_length_x, cfg.WH[0])
        view = Camera(cfg.WH, colmap_id='0000', R=R, T=T, 
                  FoVx=FovX, FoVy=FovY, depth_params=None,
                  image=None, invdepthmap=None,
                  image_name='0000', uid='0000', data_device='cuda',
                  train_test_exp=None, is_test_dataset=None, is_test_view=None,
                  K=K, normal=None, depth=None, occ_mask=None)
        prev_x = None
        relations = None
        weights = None
        image_path = f"{cfg.overlay_path}/{vis_cam_idx}/204.png"
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            N = vis_controller_points.shape[0]
            masks_ctrl_pts = []
            for i in range(n_ctrl_parts):
                mask = (cluster_labels == i)
                masks_ctrl_pts.append(torch.from_numpy(mask))
        else:
            masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0])),
            "s": (0, np.array([-0.005, 0, 0])),
            "a": (0, np.array([0, -0.005, 0])),
            "d": (0, np.array([0, 0.005, 0])),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0])),
            "k": (1, np.array([-0.005, 0, 0])),
            "j": (1, np.array([0, -0.005, 0])),
            "l": (1, np.array([0, 0.005, 0])),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005]))
        }
        self.pressed_keys = set()
        self.init_control_visualizer()


        object_colors = self.object_colors.cpu().numpy()[0]
        if object_colors.shape[0] < vis_vertices.shape[0]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            vis_vertices.shape[0] - object_colors.shape[0],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=0,
            )

        # object_pcd = o3d.geometry.PointCloud()
        # object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
        # object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
        # vis.add_geometry(object_pcd)

        controller_meshes = []
        prev_center = []
        if vis_controller_points is not None:
            # Use sphere mesh for each controller point
            for j in range(vis_controller_points.shape[0]):
                origin = vis_controller_points[j]
                origin_color = [1, 0, 0]
                controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01
                ).translate(origin)
                controller_mesh.compute_vertex_normals()
                controller_mesh.paint_uniform_color(origin_color)
                controller_meshes.append(controller_mesh)
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)

        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros((n_ctrl_parts, 3))

        ############## Temporary timer ##############
        import time
        class Timer:
            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()
                
            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()
                
            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed
            
            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0


        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            self.simulator.set_controller_interactive(prev_target, current_target)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # add the visualization code here
            # vis_vertices = x.cpu().numpy()

            # object_pcd.points = o3d.utility.Vector3dVector(vis_vertices)
            # vis.update_geometry(object_pcd)

            # if vis_controller_points is not None:
            #     for j in range(vis_controller_points.shape[0]):
            #         origin = vis_controller_points[j]
            #         controller_meshes[j].translate(origin - prev_center[j])
            #         vis.update_geometry(controller_meshes[j])
            #         prev_center[j] = origin
            # vis.poll_events()
            # vis.update_renderer()

            # frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            # frame = (frame * 255).astype(np.uint8)

            # 2. Frame initialization and setup

            frame_timer.start()

            # frame = np.ones((height, width, 3), dtype=np.uint8)
            # frame *= 255

            frame = overlay.copy()

            frame_setup_time = frame_timer.stop()  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # 3. Rendering
            render_timer.start()

            # TODO: render with gaussians and paste the image on top of the frame
            # frame = (frame / 255).astype(np.float32)
            results = render(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            render_time = render_timer.stop()
            component_times["rendering"].append(render_time)

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)
            # mask = alpha[..., 0] == 0

            # Get the mask where the pixel is white
            # mask = np.all(frame == [255, 255, 255], axis=-1)
            # frame[mask] = overlay[mask]
            # frame[mask] = alpha[mask] * frame[mask] + (1 - alpha[mask]) * overlay[mask]

            frame = self.update_frame(frame, self.pressed_keys)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # display the rendering
            # rendering = rendering.detach().cpu().numpy()
            # rendering = np.transpose(rendering, (1, 2, 0))
            # rendering = (rendering * 255).astype(np.uint8)
            # rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("./tmp.png", rendering)
            # breakpoint()

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            frame_comp_time = frame_timer.stop() + frame_setup_time  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()

            if prev_x is not None:

                prev_particle_pos = prev_x
                cur_particle_pos = x

                if relations is None:
                    relations = get_topk_indices(prev_x, K=16)                    # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(prev_particle_pos, current_pos, K=16)   # only computed in the first iteration

                interp_timer.start()

                with torch.no_grad():

                    # chunk_size = 50_000
                    # num_chunks = (len(current_pos) + chunk_size - 1) // chunk_size
                    # for j in range(num_chunks):
                    #     start = j * chunk_size
                    #     end = min((j + 1) * chunk_size, len(current_pos))
                    #     all_pos_chunk = current_pos[start:end]
                    #     all_rot_chunk = current_rot[start:end]
                    #     weights_chunk = weights[start:end]
                    #     # weights = knn_weights(prev_particle_pos, all_pos_chunk, K=16)
                    #     all_pos_chunk, all_rot_chunk, _ = interpolate_motions_feng(
                    #         bones=prev_particle_pos,
                    #         motions=cur_particle_pos - prev_particle_pos,
                    #         relations=relations,
                    #         weights=weights_chunk,
                    #         xyz=all_pos_chunk,
                    #         quat=all_rot_chunk,
                    #     )
                    #     current_pos[start:end] = all_pos_chunk
                    #     current_rot[start:end] = all_rot_chunk

                    # knn_weights_timer.start()
                    # weights = knn_weights(prev_particle_pos, current_pos, K=16)
                    # knn_weights_time = knn_weights_timer.stop()
                    # component_times["knn_weights"].append(knn_weights_time)

                    weights = calc_weights_vals_from_indices(prev_particle_pos, current_pos, weights_indices)

                    current_pos, current_rot, _ = interpolate_motions_feng_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                interp_time = interp_timer.stop()
                component_times["full_motion_interpolation"].append(interp_time)

            torch.cuda.synchronize()

            prev_x = x.clone()

            prev_target = current_target
            target_change = self.get_target_change()
            if masks_ctrl_pts is not None:
                for i in range(n_ctrl_parts):
                    if masks_ctrl_pts[i].sum() > 0:
                        current_target[masks_ctrl_pts[i]] += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
            else:
                current_target += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )

            # vis_controller_points = current_target.cpu().numpy()




            ############### Temporary timer ###############
            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)
            
            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)
            
            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][-STATS_WINDOW:]
                
                avg_fps = np.mean(fps_history)
                print(f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---")
                print(f"FPS: {avg_fps:.2f}")
                
                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")
                
                # Display individual component times
                for key in ["simulator", "rendering", "frame_compositing", "full_motion_interpolation", "knn_weights", "motion_interp"]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)")


        
        listener.stop()
