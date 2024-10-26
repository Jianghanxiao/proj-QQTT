import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
import os
from qqtt.utils import getCamera, getPcdFromRgbd

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect"
case_name = "rope_double_hand"


def get_pcd_from_data(path, frame_idx, num_cam, intrinsics, c2ws):
    total_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        color = cv2.imread(f"{path}/color/{i}/{frame_idx}.png")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0
        # import pdb
        # pdb.set_trace()
        mask = np.logical_and(depth > 0.05, depth < 1.0)
        pcd = getPcdFromRgbd(
            color,
            depth,
            intrinsics[i, 0, 0],
            intrinsics[i, 1, 1],
            intrinsics[i, 0, 2],
            intrinsics[i, 1, 2],
            mask=mask,
            is_opencv=True,
        )
        pcd.transform(c2ws[i])
        total_pcd += pcd
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
    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    exist_dir(f"{base_path}/{case_name}/pcd")

    for i in tqdm(range(frame_num)):
        pcd = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        np.savez(f"{base_path}/{case_name}/pcd/{i}.npz", points=points, colors=colors)
        
