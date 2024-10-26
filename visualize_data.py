import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
from qqtt.utils import getCamera, getPcdFromRgbd

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect"
case_name = "rope_double_hand"


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    frame_num = data["frame_num"]
    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    cameras = []
    # Visualize the cameras
    for i in range(num_cam):
        camera = getCamera(
            c2ws[i],
            intrinsics[i, 0, 0],
            intrinsics[i, 1, 1],
            intrinsics[i, 0, 2],
            intrinsics[i, 1, 2],
            z_flip=True,
            scale=0.2,
        )
        cameras += camera

    data = np.load(f"{base_path}/{case_name}/pcd/0.npz")

    pcd = o3d.geometry.PointCloud()
    mask = np.logical_and(data["points"][:, 0] < 0.3, data["points"][:, 0] > -0.3)
    pcd.points = o3d.utility.Vector3dVector(data["points"][mask])
    pcd.colors = o3d.utility.Vector3dVector(data["colors"][mask])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for camera in cameras:
        vis.add_geometry(camera)
    vis.add_geometry(pcd)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coordinate)

    view_control = vis.get_view_control()
    view_control.set_front([1, 0, -2])
    view_control.set_up([0, 0, -1])
    view_control.set_zoom(1)

    for i in range(1, frame_num):
        print(i)
        data = np.load(f"{base_path}/{case_name}/pcd/{i}.npz")
        mask = np.logical_and(data["points"][:, 0] < 0.3, data["points"][:, 0] > -0.3)
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(data["points"][mask])
        temp_pcd.colors = o3d.utility.Vector3dVector(data["colors"][mask])
        pcd.points = temp_pcd.points
        pcd.colors = temp_pcd.colors
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
