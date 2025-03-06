from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json
import open3d as o3d
import time
import torch.multiprocessing as mp
import warp as wp


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


controller_points_position = np.array(
    [
        [0.01, 0.01, 0.01],
        [-0.01, 0.01, 0.01],
        [0.01, -0.01, 0.01],
        [-0.01, -0.01, 0.01],
        [0.01, 0.01, -0.01],
        [-0.01, 0.01, -0.01],
        [0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01],
    ]
)


class PhysDynamicModule:
    def __init__(
        self,
        base_path,
        case_name,
        experiments_path,
        experiments_optimization_path,
        output_dir,
        init_pts,
        init_colors,
        init_controller_xyz,
        init_controller_rot,
        action_num,
        batch_size,
        device="cuda",
    ):
        # set the random seed, so that the results are reproducible
        seed = 42
        set_all_seeds(seed)

        self.batch_size = batch_size

        if "cloth" in case_name or "package" in case_name:
            cfg.load_from_yaml("configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/real.yaml")

        base_dir = f"{output_dir}/{case_name}"

        # Read the first-satage optimized parameters to set the indifferentiable parameters
        optimal_path = f"{experiments_optimization_path}/{case_name}/optimal_params.pkl"
        logger.info(f"Load optimal parameters from: {optimal_path}")
        assert os.path.exists(
            optimal_path
        ), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

        # Set the intrinsic and extrinsic parameters for visualization
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]

        logger.set_log_file(path=base_dir, name="inference_log")
        # self.trainers = []
        # for i in range(self.batch_size):
        #     trainer = InvPhyTrainerWarp(
        #         data_path=f"{base_path}/{case_name}/final_data.pkl",
        #         base_dir=base_dir,
        #         pure_inference_mode=True,
        #     )
        #     self.trainers.append(trainer)
        self.trainer = InvPhyTrainerWarp(
            data_path=f"{base_path}/{case_name}/final_data.pkl",
            base_dir=base_dir,
            pure_inference_mode=True,
        )

        self.device = device

        init_controller_xyz = torch.tensor(
            init_controller_xyz, dtype=torch.float, device=self.device
        )
        init_controller_rot = torch.tensor(
            init_controller_rot, dtype=torch.float32, device=self.device
        )

        self.controller_points_position = torch.tensor(
            controller_points_position, dtype=torch.float, device=self.device
        )

        self.init_controller_points = (
            torch.einsum(
                "gij,nj->gni", init_controller_rot, self.controller_points_position
            )
            + init_controller_xyz[:, None]
        )
        self.init_controller_points = torch.cat(
            list(self.init_controller_points), dim=0
        )

        # Do the alignment between the digital twin and the observations
        final_points = self.align(
            init_pts,
            init_colors,
            self.trainer.dataset.structure_points.cpu().numpy(),
            self.trainer.dataset.original_object_colors[0].cpu().numpy(),
        )

        best_model_path = glob.glob(f"{experiments_path}/{case_name}/train/best_*.pth")[
            0
        ]
        # for i in range(self.batch_size):
        #     self.trainers[i].load_model_transfer(
        #         best_model_path,
        #         self.init_controller_points.clone(),
        #         final_points.copy(),
        #         dt=5e-5,
        #     )
        self.trainer.load_model_transfer(
            best_model_path,
            self.init_controller_points.clone(),
            final_points.copy(),
            action_num=action_num,
            dt=5e-5,
        )

    def align(self, to_pts, to_colors, from_pts, from_colors):
        # Do the colored ICP registration
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(from_pts)
        source.colors = o3d.utility.Vector3dVector(from_colors)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(to_pts)
        target.colors = o3d.utility.Vector3dVector(to_colors)
        target = target.voxel_down_sample(voxel_size=0.005)
        # o3d.visualization.draw_geometries([source, target])

        # Move the source to the target
        source.translate(
            np.mean(target.points, axis=0) - np.mean(source.points, axis=0)
        )

        threshold = 0.02
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        final_points = np.array(source.transform(reg_p2p.transformation).points)

        # controller_meshes = []
        # # Use sphere mesh for each controller point
        # for j in range(self.init_controller_points.shape[0]):
        #     origin = self.init_controller_points[j]
        #     origin_color = [1, 0, 0]
        #     controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
        #         radius=0.01
        #     ).translate(origin)
        #     controller_mesh.compute_vertex_normals()
        #     controller_mesh.paint_uniform_color(origin_color)
        #     controller_meshes.append(controller_mesh)

        # o3d.visualization.draw_geometries([source, target, controller_meshes])

        return final_points

    def rollout_serialize(self, eef_xyz, eef_rot, visualize=False):
        batch_size = eef_xyz.shape[0]
        assert batch_size == self.batch_size
        all_pts = []

        for i in range(batch_size):
            with wp.ScopedTimer("rollout"):
                # Transform the controller points position based on the rotation and translation
                controller_points_array = torch.einsum(
                    "bgij,nj->bgni",
                    eef_rot[i],  # shape: (act_num, grip_num, 3, 3)
                    self.controller_points_position,  # shape: (8, 3)
                )
                controller_points_array = (
                    controller_points_array + eef_xyz[i][:, :, None, :]
                )
                controller_points_array = torch.reshape(
                    controller_points_array, [controller_points_array.shape[0], -1, 3]
                )
                controller_points_array = torch.tensor(
                    controller_points_array, dtype=torch.float, device=self.device
                ).contiguous()
                pts = self.trainer.rollout(controller_points_array, visualize=visualize)
                all_pts.append(pts)
        # all_pts = torch.stack(all_pts, dim=0)
        return all_pts


#     def rollout_parallel(self, eef_xyz, eef_rot, visualize=False):
#         result_queue = mp.Queue()

#         batch_size = eef_xyz.shape[0]
#         assert batch_size == self.batch_size
#         eef_xyz = torch.tensor(eef_xyz, dtype=torch.float, device=self.device)
#         eef_rot = torch.tensor(eef_rot, dtype=torch.float, device=self.device)

#         processes = []
#         for i in range(batch_size):
#             # 使用torch.multiprocessing.Process创建进程
#             p = mp.Process(
#                 target=rollout_worker,
#                 args=(
#                     i,
#                     eef_xyz[i],
#                     eef_rot[i],
#                     self.controller_points_position,
#                     result_queue,
#                     self.trainers[i],
#                 ),
#             )
#             processes.append(p)
#             p.start()

#         results = {}
#         for p in processes:
#             p.join()

#         while not result_queue.empty():
#             i, pts = result_queue.get()  # 获取每个子进程的结果
#             results[i] = pts

#         return results


# def rollout_worker(
#     i, eef_xyz, eef_rot, controller_points_position, trainer, result_queue
# ):
#     print(eef_rot.shape)
#     print(controller_points_position.shape)
#     controller_points_array = torch.einsum(
#         "bij,nj->bni",
#         eef_rot,  # shape: (batch_size, 3, 3)
#         controller_points_position,  # shape: (8, 3)
#     )
#     controller_points_array = controller_points_array + eef_xyz[:, None, :]
#     controller_points_array = torch.tensor(
#         controller_points_array, dtype=torch.float, device="cuda"
#     )
#     pts = trainer.rollout(controller_points_array, visualize=False)
#     result_queue.put((i, pts))


if __name__ == "__main__":
    init_pcd_path = "/home/hanxiao/Downloads/episode_0000/pcd_clean_new/000000.npz"
    init_pcd = np.load(init_pcd_path)
    init_pts = init_pcd["pts"]
    init_colors = init_pcd["colors"]

    init_controller_xyz = np.array(
        [
            [
                3.044547887605847380e-01,
                2.841108186878805730e-01,
                -1.315379715678757083e-02,
            ]
        ]
    )
    init_controller_rot = np.array(
        [
            [
                [
                    -6.480154314145059047e-02,
                    9.962636060017686646e-01,
                    -5.709279606780431893e-02,
                ],
                [
                    -9.961174236442396079e-01,
                    -6.799639288895997780e-02,
                    -5.591573004489115012e-02,
                ],
                [
                    -5.958891103930036293e-02,
                    5.324770333491749691e-02,
                    9.968018076682580997e-01,
                ],
            ]
        ]
    )

    batch_size = 5
    action_num = 143

    dynamic_module = PhysDynamicModule(
        base_path="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types",
        case_name="single_lift_rope",
        experiments_path="experiments",
        experiments_optimization_path="experiments_optimization",
        output_dir="temp_experiments",
        init_pts=init_pts,
        init_colors=init_colors,
        init_controller_xyz=init_controller_xyz,
        init_controller_rot=init_controller_rot,
        action_num=action_num,
        batch_size=batch_size,
        device="cuda",
    )

    controller_xyzs = []
    controller_rots = []
    for i in range(142):
        path = f"/home/hanxiao/Downloads/episode_0000/robot/{(i+1):06d}.txt"
        with open(path, "r") as f:
            lines = f.readlines()
            controller_xyz = np.array(
                [float(x) for x in lines[0].strip().split(" ")], dtype=np.float32
            )
            controller_rot = np.array(
                [float(x) for x in lines[1].strip().split(" ")]
                + [float(x) for x in lines[2].strip().split(" ")]
                + [float(x) for x in lines[3].strip().split(" ")],
                dtype=np.float32,
            ).reshape(3, 3)
            controller_xyzs.append(controller_xyz)
            controller_rots.append(controller_rot)

    controller_xyzs = np.array(controller_xyzs)
    controller_rots = np.array(controller_rots)

    print("Finish initialization!!!!!!!!!!!!!!!!!!!!")
    # Batch_size * action_num * num_gripper * 3
    results = dynamic_module.rollout_serialize(
        torch.tensor(
            [controller_xyzs[:, None, :]] * batch_size,
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [controller_rots[:, None, :]] * batch_size,
            dtype=torch.float32,
            device="cuda",
        ),
        visualize=False,
    )
