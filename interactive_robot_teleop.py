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
import cv2


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


# Original Reference: proj-QQTT/physics_dynamic_module.py
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
        episode_name,
        device="cuda",
    ):
        # set the random seed, so that the results are reproducible
        seed = 42
        set_all_seeds(seed)

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

        # Set the intrinsic and extrinsic parameters for visualization (TODO: modify this part to take multiple cameras)
        # with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        #     c2ws = pickle.load(f)
        # w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        # cfg.c2ws = np.array(c2ws)
        # cfg.w2cs = np.array(w2cs)
        # with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        #     data = json.load(f)
        # cfg.intrinsics = np.array(data["intrinsics"])
        # cfg.WH = data["WH"]

        cfg.episode_path = f"/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/{episode_name}"
        calibration_dir = f"{cfg.episode_path}/calibration/"
        h, w = 480, 848
        intr = np.load(calibration_dir + 'intrinsics.npy')
        rvec = np.load(calibration_dir + 'rvecs.npy')
        tvec = np.load(calibration_dir + 'tvecs.npy')
        R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
        T = [tvec[i, :, 0] for i in range(tvec.shape[0])]
        extrs = np.zeros((len(R), 4, 4)).astype(np.float32)
        for i in range(len(R)):
            extrs[i, :3, :3] = R[i]
            extrs[i, :3, 3] = T[i]
            extrs[i, 3, 3] = 1
        cfg.w2cs = extrs
        cfg.intrinsics = intr
        cfg.WH = (w, h)
        

        logger.set_log_file(path=base_dir, name="inference_log")

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
            + init_controller_xyz
        )
        self.init_controller_points = torch.cat(
            list(self.init_controller_points), dim=0
        )

        # Do the alignment between the digital twin and the observations
        final_points, align_translation, align_p2p_transform = self.align(
            init_pts,
            init_colors,
            self.trainer.dataset.structure_points.cpu().numpy(),
            self.trainer.dataset.original_object_colors[0].cpu().numpy(),
        )
        self.align_translation = align_translation
        self.align_p2p_transform = align_p2p_transform


        best_model_path = glob.glob(f"{experiments_path}/{case_name}/train/best_*.pth")[
            0
        ]

        # Use `SpringMassSystemWarp` instead of `SpringMassSystemWarpAccelerate`
        self.trainer.load_model_transfer_no_acc(
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

        align_translation = np.mean(target.points, axis=0) - np.mean(source.points, axis=0)

        # Move the source to the target
        source.translate(
            np.mean(target.points, axis=0) - np.mean(source.points, axis=0)  # translation (1x3 np.array)
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
        # o3d.visualization.draw_geometries([source, target])

        align_p2p_transform = reg_p2p.transformation  # transformation (4x4 np.array)

        # # visualize the aligned point cloud
        # source_transformed = o3d.geometry.PointCloud()
        # source_transformed.points = o3d.utility.Vector3dVector(np.asarray(source.points))
        # source_transformed.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))
        # source_transformed.transform(reg_p2p.transformation)
        
        # # Visualize the result
        # print("Showing point clouds after ICP alignment")
        # self.visualize_point_clouds([source_transformed, target], ["Source (Aligned)", "Target"])


        return final_points, align_translation, align_p2p_transform
    
    
    def visualize_point_clouds(self, geometries, window_names=None):
        """
        Visualize multiple point clouds with custom names.
        
        Args:
            geometries: List of point cloud geometries
            window_names: List of names for each point cloud
        """
        if window_names is None:
            window_names = [f"Point Cloud {i}" for i in range(len(geometries))]
        
        # Visualize combined view
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Combined Point Clouds")
        
        # Assign different colors to differentiate between point clouds
        colors = [[1, 0.706, 0], [0, 0.651, 0.929]]  # Orange for source, blue for target
        
        for i, (geom, name) in enumerate(zip(geometries, window_names)):
            # Use the geometry directly as we're only adding it to the visualizer
            geom_copy = geom
            
            # Add to combined visualizer
            vis.add_geometry(geom_copy)
        
        # Set viewing options and run the visualizer
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
        opt.point_size = 2.0
        
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":

    episode_name = "episode_0001"

    init_pcd_path = f"/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/{episode_name}/pcd_clean_new/000000.npz"
    init_pcd = np.load(init_pcd_path)
    init_pts = init_pcd["pts"]
    init_colors = init_pcd["colors"]

    # init_controller_xyz = np.array(
    #     [
    #         [
    #             3.044547887605847380e-01,
    #             2.841108186878805730e-01,
    #             -1.315379715678757083e-02,
    #         ]
    #     ]
    # )
    # init_controller_rot = np.array(
    #     [
    #         [
    #             [
    #                 -6.480154314145059047e-02,
    #                 9.962636060017686646e-01,
    #                 -5.709279606780431893e-02,
    #             ],
    #             [
    #                 -9.961174236442396079e-01,
    #                 -6.799639288895997780e-02,
    #                 -5.591573004489115012e-02,
    #             ],
    #             [
    #                 -5.958891103930036293e-02,
    #                 5.324770333491749691e-02,
    #                 9.968018076682580997e-01,
    #             ],
    #         ]
    #     ]
    # )

    rollout_dir = os.path.join("/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT", episode_name, "robot")
    controller_xyzs = []
    controller_rots = []
    for i in range(len(os.listdir(rollout_dir))):
        path = os.path.join(rollout_dir, f"{(i):06d}.txt")
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

    init_controller_xyz = controller_xyzs[0].reshape(1, 3)
    init_controller_rot = controller_rots[0].reshape(1, 3, 3)
    action_num = controller_xyzs.shape[0]

    action_num = 143
    case_name = "single_push_rope"

    dynamic_module = PhysDynamicModule(
        base_path="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian_data",
        case_name=case_name,
        experiments_path="experiments",
        experiments_optimization_path="experiments_optimization",
        output_dir="temp_experiments",
        init_pts=init_pts,
        init_colors=init_colors,
        init_controller_xyz=init_controller_xyz,
        init_controller_rot=init_controller_rot,
        action_num=action_num,
        episode_name=episode_name,
        device="cuda",
    )

    print("Finish initialization!!!!!!!!!!!!!!!!!!!!")

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]

    exp_name = 'init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'
    gaussians_path = f"/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/gaussian_splatting/output/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    dynamic_module.trainer.interactive_robot_teleop(
        gaussians_path,
        dynamic_module.align_translation,
        dynamic_module.align_p2p_transform,
        dynamic_module.controller_points_position,
    )
