import glob
import pickle
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from qqtt.utils import visualize_pc, cfg

# prediction_dir = (
#     "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments_transfer"
# )
# prediction_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/gnn_test/different_types/output"
# prediction_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/exp_results/GNN/different_types_gnn"
prediction_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments"
base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"

dir_names = glob.glob(f"{prediction_dir}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}")
    # if case_name != "double_stretch_sloth":
    #     continue

    # Read the trajectory data
    with open(f"{dir_name}/inference.pkl", "rb") as f:
        vertices = pickle.load(f)

    # Read the controller points for visualization
    with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
        data = pickle.load(f)

    controller_points = data["controller_points"]
    object_points = data["object_points"]

    # Get the rainbow color from the original object points
    # Get the rainbow color for the object_colors
    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    rainbow_colors = torch.tensor(
        rainbow_colors, dtype=torch.float32, device="cuda"
    )
    # Make the same rainbow color for each frame
    object_colors = rainbow_colors.repeat(object_points.shape[0], 1, 1)

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
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    visualize_pc(
        vertices,
        object_colors,
        controller_points=controller_points,
        visualize=False,
        save_video=True,
        save_path=f"{dir_name}/inference.mp4",
    )
