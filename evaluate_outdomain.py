import glob
import pickle
import json
import torch
import csv
import numpy as np
from scipy.spatial import KDTree
from pytorch3d.loss import chamfer_distance

prediction_dir = (
    "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments_out_domain"
)
base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
output_file = "results/final_outdomain_results.csv"


# TODO: the tracking error is meaningless for now
def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    from_data,
    to_data,
):
    from_object_points = from_data["object_points"]
    from_object_visibilities = from_data["object_visibilities"]
    from_object_motions_valid = from_data["object_motions_valid"]
    from_num_original_points = from_object_points.shape[1]
    from_num_surface_points = (
        from_num_original_points + from_data["surface_points"].shape[0]
    )

    to_object_points = to_data["object_points"]
    to_object_visibilities = to_data["object_visibilities"]
    to_object_motions_valid = to_data["object_motions_valid"]
    to_num_original_points = to_object_points.shape[1]
    to_num_surface_points = to_num_original_points + to_data["surface_points"].shape[0]

    chamfer_errors = []
    track_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(from_object_points, torch.Tensor):
        from_object_points = torch.tensor(from_object_points, dtype=torch.float32)
    if not isinstance(from_object_visibilities, torch.Tensor):
        from_object_visibilities = torch.tensor(
            from_object_visibilities, dtype=torch.bool
        )
    if not isinstance(from_object_motions_valid, torch.Tensor):
        from_object_motions_valid = torch.tensor(
            from_object_motions_valid, dtype=torch.bool
        )
    if not isinstance(to_object_points, torch.Tensor):
        to_object_points = torch.tensor(to_object_points, dtype=torch.float32)
    if not isinstance(to_object_visibilities, torch.Tensor):
        to_object_visibilities = torch.tensor(to_object_visibilities, dtype=torch.bool)
    if not isinstance(to_object_motions_valid, torch.Tensor):
        to_object_motions_valid = torch.tensor(to_object_motions_valid, dtype=torch.bool)

    # In the first frame, locate the nearest point for each point in to case
    kdtree = KDTree(vertices[0].numpy())
    _, indices = kdtree.query(to_object_points[0].numpy())
    indices = torch.tensor(indices, dtype=torch.int)
    track_vertices = vertices[:, indices]

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        gt_object_points = to_object_points[frame_idx]
        gt_object_visibilities = to_object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        gt_object_motions_valid = to_object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = gt_object_points[gt_object_visibilities]
        chamfer_x = x[:from_num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        # Compute the tracking loss for the object points
        gt_track_points = gt_object_points[gt_object_motions_valid]
        pred_x = track_vertices[frame_idx][gt_object_motions_valid]
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

    with open("out_domain.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    all_cases = []
    for row in data:
        all_cases.append((row[0], row[1]))
        all_cases.append((row[1], row[0]))

    for from_case, to_case in all_cases:
        print(f"Processing {from_case} to {to_case}")
        exp_name = f"{from_case}_to_{to_case}"

        # Read the trajectory data
        with open(f"{prediction_dir}/{exp_name}/inference.pkl", "rb") as f:
            vertices = pickle.load(f)

        # Read the GT object points and masks
        with open(f"{base_path}/{from_case}/final_data.pkl", "rb") as f:
            from_data = pickle.load(f)

        with open(f"{base_path}/{to_case}/final_data.pkl", "rb") as f:
            to_data = pickle.load(f)

        # read the train/test split
        with open(f"{base_path}/{to_case}/split.json", "r") as f:
            split = json.load(f)
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        assert (
            test_frame == vertices.shape[0]
        ), f"Test frame {test_frame} != {vertices.shape[0]}"

        # Do the statistics on all data, from 2nd frame
        results = evaluate_prediction(
            1,
            test_frame,
            vertices,
            from_data,
            to_data,
        )

        writer.writerow(
            [
                exp_name,
                results["frame_len"],
                results["chamfer_error"],
                results["track_error"],
            ]
        )
    file.close()
