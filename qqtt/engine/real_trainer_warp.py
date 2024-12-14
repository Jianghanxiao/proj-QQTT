from qqtt.data import RealData
from qqtt.utils import logger, visualize_pc_real, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp


class RealInvPhyTrainerWarp:
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
        self.structure_points = self.dataset.structure_points
        self.num_original_points = self.dataset.num_original_points
        self.num_surface_points = self.dataset.num_surface_points
        self.num_all_points = self.dataset.num_all_points

        # Initialize the vertices, springs, rest lengths and masses
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
        ) = self._init_start(
            self.structure_points,
            self.controller_points[0],
            radius=cfg.radius,
            max_neighbours=cfg.max_neighbours,
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
            # collide_elas=cfg.init_collide_elas,
            # collide_fric=cfg.init_collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            # collide_object_elas=cfg.collide_object_elas,
            # collide_object_fric=cfg.collide_object_fric,
            # init_masks=self.init_masks,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=True,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
        )

        self.optimizer = torch.optim.Adam(
            [wp.to_torch(self.simulator.wp_spring_Y)], lr=cfg.base_lr, betas=(0.9, 0.99)
        )

        if "debug" not in cfg.run_name:
            wandb.init(
                # set the wandb project where this run will be logged
                project="real_data",
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
        radius=0.01,
        max_neighbours=20,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
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
                    points[i], radius, max_neighbours
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

            # Connec the springs between the controller points and the object points
            num_object_points = len(points)
            points = np.concatenate([points, controller_points], axis=0)
            for i in range(len(controller_points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    controller_points[i], radius * 2, max_neighbours * 4
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
            total_chamfer_loss = 0.0
            total_track_loss = 0.0
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, self.dataset.frame_len)):
                    self.simulator.set_controller_target(j)

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        with self.simulator.tape:
                            self.simulator.step()
                            self.simulator.calculate_loss()
                        self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()

                    chamfer_loss = wp.to_torch(
                        self.simulator.chamfer_loss, requires_grad=False
                    )
                    track_loss = wp.to_torch(
                        self.simulator.track_loss, requires_grad=False
                    )
                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()
                    total_chamfer_loss += chamfer_loss.item()
                    total_track_loss += track_loss.item()

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

            total_loss /= self.dataset.frame_len - 1
            total_chamfer_loss /= self.dataset.frame_len - 1
            total_track_loss /= self.dataset.frame_len - 1
            # total_acc_loss /= self.dataset.frame_len - 2
            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": total_chamfer_loss,
                    "track_loss": total_track_loss,
                    # "acc_loss": total_acc_loss,
                    # "collide_else": self.simulator.collide_elas.item(),
                    # "collide_fric": self.simulator.collide_fric.item(),
                    # "collide_object_elas": self.simulator.collide_object_elas.item(),
                    # "collide_object_fric": self.simulator.collide_object_fric.item(),
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
                # TODO: Save other parameters
                cur_model = {
                    "epoch": i,
                    "spring_Y": wp.to_torch(
                        self.simulator.wp_spring_Y, requires_grad=False
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

    def visualize_sim(
        self, save_only=True, video_path=None, springs=None, spring_params=None
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
                self.simulator.set_controller_target(i)
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

        if not save_only:
            visualize_pc_real(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc_real(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )
