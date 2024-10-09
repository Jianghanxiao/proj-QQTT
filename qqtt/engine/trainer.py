from qqtt.data import SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt import SpringMassSystem
import open3d as o3d
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
import wandb
import os
from pytorch3d.loss import chamfer_distance


class InvPhyTrainer:
    def __init__(
        self, data_path, base_dir, mask_path=None, velocity_path=None, device="cuda:0"
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        # If match is True, use mse loss, otherwise use chamfer loss
        self.match = cfg.match
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
        self.optimizer = torch.optim.Adam(
            self.simulator.parameters(), lr=cfg.base_lr, betas=(0.9, 0.99)
        )
        if "debug" not in cfg.run_name:
            wandb.init(
                # set the wandb project where this run will be logged
                project="billiard_fix",
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

    def train(self, start_epoch=-1):
        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
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
                loss = self.compute_points_loss(self.dataset.data[j], x, match=self.match)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.simulator.detach()
                total_loss += loss.item()
                # if torch.isnan(self.simulator.spring_Y.grad).sum() > 0:
                #     print(torch.isnan(self.simulator.spring_Y.grad).sum())
                #     print("Nan detected for spring_Y")
                #     import pdb
                #     pdb.set_trace()
            total_loss /= self.dataset.frame_len - 1
            wandb.log(
                {
                    "loss": total_loss,
                    "collide_else": self.simulator.collide_elas.item(),
                    "collide_fric": self.simulator.collide_fric.item(),
                    "collide_object_elas": self.simulator.collide_object_elas.item(),
                    "collide_object_fric": self.simulator.collide_object_fric.item(),
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
                cur_model = {
                    "epoch": i,
                    "model_state_dict": self.simulator.state_dict(),
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

    def compute_points_loss(self, gt, x, match=True):
        if match:
            # Compute the mse loss between the ground truth and the predicted points
            return smooth_l1_loss(x, gt, beta=1.0, reduction="mean")
        else:
            # Computer the chamfer loss between the ground truth and the predicted points
            return chamfer_distance(x.unsqueeze(0), gt.unsqueeze(0))[0]

    def test(self, model_path, normalization_factor=1e5):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)
        self.simulator.load_state_dict(checkpoint["model_state_dict"])
        self.simulator.to(cfg.device)

        springs = self.simulator.springs.cpu().numpy()
        spring_params = (
            torch.exp(self.simulator.spring_Y).detach().cpu().numpy() / normalization_factor
        )
        self.visualize_sim(
            save_only=False,
            springs=springs,
            spring_params=spring_params,
        )
        video_path = f"{cfg.base_dir}/test_visual.mp4"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            springs=springs,
            spring_params=spring_params,
        )

    def resume_train(self, model_path):
        # Load the model
        checkpoint = torch.load(model_path, map_location=cfg.device)
        epoch = checkpoint["epoch"]
        logger.info(f"Continue training with model from {model_path} at epoch {epoch}")
        self.simulator.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.simulator.to(cfg.device)

        self.train(epoch)

    def visualize_sim(
        self, save_only=True, video_path=None, springs=None, spring_params=None
    ):
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
            
            # if True:
            #     left_index = []
            #     pcds = self.init_vertices.cpu().numpy()
            #     for spring in springs:
            #         p1 = pcds[spring[0]]
            #         p2 = pcds[spring[1]]
            #         if p1[1] < 0.49:
            #             left_index.append(True)
            #         else:
            #             left_index.append(False)
            #     left_index = np.array(left_index)
            #     springY_left = spring_params[left_index]
            #     springY_right = spring_params[~left_index]

                    
            #     import pdb
            #     pdb.set_trace()

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
                    springs=springs,
                    spring_params=spring_params,
                )
            else:
                assert video_path is not None, "Please provide the video path to save"
                visualize_pc(
                    vertices,
                    visualize=False,
                    save_video=True,
                    save_path=video_path,
                    springs=springs,
                    spring_params=spring_params,
                )

    def visualize_gt(self):
        # Visualize the ground truth data
        self.dataset.visualize_data()
