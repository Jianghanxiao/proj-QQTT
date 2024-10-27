import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
import os
from qqtt.utils import getCamera, getPcdFromRgbd

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect"
# case_name = "rope_double_hand"
case_name = "test_rope"
OBJECT_NAME = "rope"
CONTROLLER_NAME = "hand"


def get_pcd_from_data(path, frame_idx, num_cam, intrinsics, c2ws, mask_info):
    total_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        color = cv2.imread(f"{path}/color/{i}/{frame_idx}.png")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0

        final_mask = np.logical_and(depth > 0.05, depth < 2.0)
        pcd = getPcdFromRgbd(
            color,
            depth,
            intrinsic=intrinsics[i],
            mask=final_mask,
            is_opencv=True,
        )
        # if i== 2 :
        # pcd.paint_uniform_color(list(np.random.rand(3)))
        pcd.transform(c2ws[i])
        total_pcd += pcd
    coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([total_pcd, coordinates])
    return total_pcd


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    frame_num = data["frame_num"]
    print(data["serial_numbers"])
    # import pdb
    # pdb.set_trace()
    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))
    # Load the mask metadata
    mask_info = {}
    # for i in range(num_cam):
    #     with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
    #         data = json.load(f)
    #     mask_info[i] = {}
    #     for key, value in data.items():
    #         if value == OBJECT_NAME:
    #             mask_info[i]["object"] = int(key)
    #         if value == CONTROLLER_NAME:
    #             if "controller" in mask_info[i]:
    #                 mask_info[i]["controller"].append(int(key))
    #             else:
    #                 mask_info[i]["controller"] = [int(key)]

    exist_dir(f"{base_path}/{case_name}/pcd")
    exist_dir(f"{base_path}/{case_name}/pcd/object")
    exist_dir(f"{base_path}/{case_name}/pcd/controller")

    for i in tqdm(range(frame_num)):
        object_pcd, controller_pcd = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws, mask_info
        )
        object_points = np.asarray(object_pcd.points)
        object_colors = np.asarray(object_pcd.colors)
        np.savez(
            f"{base_path}/{case_name}/pcd/object/{i}.npz",
            points=object_points,
            colors=object_colors,
        )

        controller_points = np.asarray(controller_pcd.points)
        controller_colors = np.asarray(controller_pcd.colors)
        np.savez(
            f"{base_path}/{case_name}/pcd/controller/{i}.npz",
            points=controller_points,
            colors=controller_colors,
        )
