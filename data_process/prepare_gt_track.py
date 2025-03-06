# Read the manual annotated 2D keypoint and find the nearest reliable 3D points
import os
import glob
import pickle
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import open3d as o3d

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
gt_track_path = (
    "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_gt_track"
)
VIS = True

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    # Load the GT track data
    with open(f"{gt_track_path}/{case_name}/0_tracking.pkl", "rb") as f:
        gt_track = pickle.load(f)

    if VIS:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)

    gt_track_3d = []

    for i in range(gt_track.shape[0]):
        # Load the point cloud data for the first viewpoint
        pcd_path = f"{base_path}/{case_name}/pcd/{i}.npz"
        mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
        data = np.load(pcd_path)
        with open(mask_path, "rb") as f:
            processed_masks = pickle.load(f)
        points = data["points"][0]
        colors = data["colors"][0]
        mask = processed_masks[i][0]["object"]

        # new_match, matching_points = select_point(points, gt_track[i], mask)
        track_mask = (gt_track[i] != -1).all(axis=1)
        matching_points = []

        # import pdb
        # pdb.set_trace()
        for j in range(gt_track.shape[1]):
            if gt_track[i, j, 0] != -1:
                try:
                    if mask[gt_track[i, j, 1], gt_track[i, j, 0]]:
                        matching_points.append(
                            points[gt_track[i, j, 1], gt_track[i, j, 0]]
                        )
                    else:
                        matching_points.append([np.nan, np.nan, np.nan])
                except:
                    import pdb

                    pdb.set_trace()
            else:
                matching_points.append([np.nan, np.nan, np.nan])
        matching_points = np.array(matching_points)
        gt_track_3d.append(matching_points)
        if VIS:
            if i == 0:
                object_pcd = o3d.geometry.PointCloud()
                object_pcd.points = o3d.utility.Vector3dVector(points[mask])
                object_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
                vis.add_geometry(object_pcd)

                keypoint_pcd = o3d.geometry.PointCloud()
                keypoint_pcd.points = o3d.utility.Vector3dVector(
                    matching_points[track_mask]
                )
                keypoint_pcd.paint_uniform_color([1, 0, 0])
                vis.add_geometry(keypoint_pcd)
            else:
                object_pcd.points = o3d.utility.Vector3dVector(points[mask])
                object_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
                vis.update_geometry(object_pcd)

                keypoint_pcd.points = o3d.utility.Vector3dVector(
                    matching_points[track_mask]
                )
                keypoint_pcd.paint_uniform_color([1, 0, 0])
                vis.update_geometry(keypoint_pcd)

            vis.poll_events()
            vis.update_renderer()
    gt_track_3d = np.array(gt_track_3d)
    with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "wb") as f:
        pickle.dump(gt_track_3d, f)
