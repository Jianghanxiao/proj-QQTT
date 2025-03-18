import glob
import os
import open3d as o3d
import numpy as np
import json
import csv
import pickle
from scipy.spatial import KDTree
from pytorch3d.loss import chamfer_distance
import torch

prediction_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/baselines/Spring-Gaus/exp"
base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
output_file = "results/final_indomain_SG.csv"

def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    chamfer_errors = []
    track_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        # Compute the tracking loss for the object points
        gt_track_points = current_object_points[current_object_motions_valid]
        pred_x = x[:num_original_points][current_object_motions_valid]
        track_error = torch.mean(((pred_x - gt_track_points) ** 2).sum(-1) ** 0.5)

        chamfer_errors.append(chamfer_error.item())
        track_errors.append(track_error.item())

    chamfer_errors = np.array(chamfer_errors)
    track_errors = np.array(track_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
        "track_error": np.mean(track_errors),
    }

    return results


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
        try:
            case_name = dir_name.split("/")[-1]
            # if not os.path.exists(f"{prediction_dir}/{case_name}/evaluations/0"):
            #     continue
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
            vertices = np.array(sg_points)

            # Read the GT object points and masks
            with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
                data = pickle.load(f)

            object_points = data["object_points"]
            object_visibilities = data["object_visibilities"]
            object_motions_valid = data["object_motions_valid"]
            num_original_points = object_points.shape[1]
            num_surface_points = num_original_points + data["surface_points"].shape[0]

            # Do the statistics on train split, only evalaute from the 2nd frame
            results_train = evaluate_prediction(
                1,
                train_frame,
                vertices,
                object_points,
                object_visibilities,
                object_motions_valid,
                num_original_points,
                num_surface_points,
            )
            results_test = evaluate_prediction(
                train_frame,
                test_frame,
                vertices,
                object_points,
                object_visibilities,
                object_motions_valid,
                num_original_points,
                num_surface_points,
            )

            writer.writerow(
                [
                    case_name,
                    results_train["frame_len"],
                    results_train["chamfer_error"],
                    results_train["track_error"],
                    results_test["frame_len"],
                    results_test["chamfer_error"],
                    results_test["track_error"],
                ]
            )
        except:
            pass
    file.close()

