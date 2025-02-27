import glob
import os
import open3d as o3d
import numpy as np
import json
import csv
import pickle
from scipy.spatial import KDTree
from pytorch3d.loss import chamfer_distance

prediction_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/baselines/Spring-Gaus/exp"
base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
output_file = "results/final_indomain_SG.csv"

if __name__ == "__main__":
    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)

    writer.writerow(
        [
            "Case Name",
            "Frame Num",
            "Chamfer Error",
            "Track Error",
        ]
    )

    dir_names = glob.glob(f"{base_path}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        if not os.path.exists(f"{prediction_dir}/{case_name}/evaluations/0"):
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        frame_len = split["frame_len"]
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
            data = pickle.load(f)

        sg_points = []

        for frame_idx in range(frame_len):
            pcd_path = f"{prediction_dir}/{case_name}/evaluations/0/simulate_pred/pred_{frame_idx}.ply"
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            sg_points.append(points)
        sg_points = np.array(sg_points)
        import pdb

        pdb.set_trace()

        # train_results = evaluate_prediction(
        #     1,
        #     train_frame,
        #     sg_points,
        #     data,
        # )
