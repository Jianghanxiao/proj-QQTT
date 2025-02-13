import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import pickle
import trimesh
import cv2
import json
import torch
import os
from utils.align_util import (
    render_multi_images,
    as_mesh,
    get_bbox,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
)
from match_pairs import image_pair_matching
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name

if __name__ == "__main__":
    cam_idx = 0
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    mesh_path = f"{base_path}/{case_name}/shape/object.glb"
    # Get the mask index of the object
    with open(f"{base_path}/{case_name}/mask/mask_info_{cam_idx}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_img_path = f"{base_path}/{case_name}/mask/{cam_idx}/{obj_idx}/0.png"
    # Load the metadata
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsic = np.array(data["intrinsics"])[cam_idx]

    # Load the shape prior
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = as_mesh(mesh)

    # Load and process the image to get a cropped version for easy superglue
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # Get mask bounding box, larger than the original bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    bbox = np.argwhere(mask_img > 0.8 * 255)
    bbox = (
        np.min(bbox[:, 1]),
        np.min(bbox[:, 0]),
        np.max(bbox[:, 1]),
        np.max(bbox[:, 0]),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2),
    )
    # Make sure the bounding box is within the image
    bbox = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(raw_img.shape[1], bbox[2]),
        min(raw_img.shape[0], bbox[3]),
    )
    # Get the masked cropped image used for superglue
    crop_img = raw_img.copy()
    mask_bool = mask_img > 0
    crop_img[~mask_bool] = 0
    crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
    print("rendering objects...")
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]

    # Use superglue to match the features
    print("matching features...")
    best_pose, match_result = image_pair_matching(
        grays, crop_img, "./test_output", viz_best=True
    )
    chosen_pose = camera_poses[best_pose].cpu().numpy()
    print("best_pose", np.array2string(chosen_pose, separator=", "))
    print("matched point number", np.sum(match_result["matches"] > -1))
    # cv2.imshow("Reference image", colors[best_pose])
    # cv2.waitKey(0)

    valid_matches = match_result["matches"] > -1
    image_points = match_result["keypoints0"][valid_matches]
    world_points, valid_mask = project_2d_to_3d(
        image_points, depths[best_pose], camera_intrinsics, chosen_pose
    )
    image_points = image_points[valid_mask]

    # Visualize the correpsonded points on the mesh and on the image
    plot_mesh_with_points(mesh, world_points, "test1.png")
    plot_image_with_points(depths[best_pose], image_points, "test2.png")

    # Process matched points on original raw frame
    # - keypoints1 is of the original image
    # - match_points_on_raw: 2D points of keypoints1 in pixel coordinates
    match_points_on_mask = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    match_points_on_mask = match_points_on_mask[valid_mask]

    match_points_on_raw = match_points_on_mask + np.array([bbox[0], bbox[1]])
    plot_image_with_points(raw_img, match_points_on_raw, "test_3.png")
