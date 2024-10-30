import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
import os
import glob
import cv2
import pickle

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect"
case_name = "rope_double_hand"
OBJECT_NAME = "twine"
CONTROLLER_NAME = "hand"


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# Deprecated: throwing away the whole trajectory doesn't make that much sense
def strict_filter_track(track_path, mask_path, frame_num, num_cam):
    with open(f"{mask_path}/processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(f)
    
    track_data = {}
    for i in tqdm(range(num_cam)):
        current_track_data = np.load(f"{track_path}/{i}.npz")
        # Filter out the track data
        tracks = current_track_data["tracks"]
        tracks = np.round(tracks).astype(int)
        visibility = current_track_data["visibility"]
        assert tracks.shape[0] == frame_num
        num_points = np.shape(tracks)[1]

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i]["object"]
        track_object_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_object_idx[j] = object_mask[tracks[0, j, 0], tracks[0, j, 1]]
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i]["controller"]
        track_controller_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_controller_idx[j] = controller_mask[tracks[0, j, 0], tracks[0, j, 1]]

        # Filter out bad tracking in other frames
        for frame_idx in range(1, frame_num):
            # Filter based on object_mask
            object_mask = processed_masks[frame_idx][i]["object"]
            for j in range(num_points):
                # if visibility[frame_idx, j] == 0:
                #     track_object_idx[j] = 0
                if track_object_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not object_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        track_object_idx[j] = 0
            # Filter based on controller_mask
            controller_mask = processed_masks[frame_idx][i]["controller"]
            for j in range(num_points):
                # if visibility[frame_idx, j] == 0:
                #     track_controller_idx[j] = 0
                if track_controller_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not controller_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        track_controller_idx[j] = 0

        final_valid_idx = np.logical_or(track_object_idx, track_controller_idx)
        track_data[i] = {}
        track_data[i]["tracks"] = tracks
        track_data[i]["visibility"] = visibility
        track_data[i]["track_object_idx"] = track_object_idx
        track_data[i]["track_controller_idx"] = track_controller_idx
        track_data[i]["final_valid_idx"] = final_valid_idx

    return track_data


def process_pcd_mask(
    frame_idx,
    pcd_path,
    mask_path,
    num_cam,
    track_data,
):
    # Load the pcd data
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]
    colors = data["colors"]
    masks = data["masks"]

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for i in range(num_cam):
        current_track_data = track_data[i]
        tracks = current_track_data["tracks"][frame_idx]
        track_object_idx = current_track_data["track_object_idx"]
        valid_pixels = tracks[np.where(track_object_idx == 1)]

        object_points = points[i][valid_pixels[:, 0], valid_pixels[:, 1]]
        object_colors = colors[i][valid_pixels[:, 0], valid_pixels[:, 1]]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Load the controller mask
        track_controller_idx = current_track_data["track_controller_idx"]
        valid_pixels = tracks[np.where(track_controller_idx == 1)]

        controller_points = points[i][valid_pixels[:, 0], valid_pixels[:, 1]]
        controller_colors = colors[i][valid_pixels[:, 0], valid_pixels[:, 1]]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # # Apply the outlier removal
    # cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    # object_pcd = object_pcd.select_by_index(ind)

    # cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    # controller_pcd = controller_pcd.select_by_index(ind)

    # controller_pcd.paint_uniform_color([1, 0, 0])

    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])
    return object_pcd, controller_pcd




if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"
    track_path = f"{base_path}/{case_name}/cotracker"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))


    # # # No filter on the tracking part
    # # Strict Filter: filter trajectories from the vp if the point is not always in the mask (filter the points disappering from the video)
    track_data = strict_filter_track(track_path, mask_path, frame_num, num_cam)
    # with open("test.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # # Load the pkl data
    # with open("test.pkl", "rb") as f:
    #     track_data = pickle.load(f)


    vis = o3d.visualization.Visualizer()
    vis.create_window()

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i,
            pcd_path,
            mask_path,
            num_cam,
            track_data,
        )
        if i == 0:
            object_pcd = temp_object_pcd
            controller_pcd = temp_controller_pcd
            vis.add_geometry(object_pcd)
            vis.add_geometry(controller_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
            # object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
            controller_pcd.points = o3d.utility.Vector3dVector(
                temp_controller_pcd.points
            )
            # controller_pcd.colors = o3d.utility.Vector3dVector(
            #     temp_controller_pcd.colors
            # )
            vis.update_geometry(object_pcd)
            vis.update_geometry(controller_pcd)
            vis.poll_events()
            vis.update_renderer()
