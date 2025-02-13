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
    render_image,
    as_mesh,
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


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def pose_selection_render_superglue(
    raw_img, fov, mesh_path, mesh, crop_img, output_dir
):
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
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
    best_idx, match_result = image_pair_matching(
        grays, crop_img, output_dir, viz_best=True
    )
    print("matched point number", np.sum(match_result["matches"] > -1))

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    return best_color, best_depth, best_pose, match_result, camera_intrinsics


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

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    if not os.path.exists(f"{base_path}/{case_name}/shape/matching/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
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

        existDir(f"{base_path}/{case_name}/shape/matching")
        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                mesh_path,
                intrinsic,
                mesh,
                crop_img,
                output_dir=f"{base_path}/{case_name}/shape/matching",
            )
        )
        with open(f"{base_path}/{case_name}/shape/matching/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )
    else:
        with open(f"{base_path}/{case_name}/shape/matching/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )

    # Get the projected 3D matching points on the mesh
    valid_matches = match_result["matches"] > -1
    render_matching_points = match_result["keypoints0"][valid_matches]
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]
    # Get the matching points on the raw image
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    raw_matching_points_box = raw_matching_points_box[valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

    # Do visualization for the matching
    plot_mesh_with_points(
        mesh,
        mesh_matching_points,
        f"{base_path}/{case_name}/shape/matching/mesh_matching.png",
    )
    plot_image_with_points(
        best_depth,
        render_matching_points,
        f"{base_path}/{case_name}/shape/matching/render_matching.png",
    )
    plot_image_with_points(
        raw_img,
        raw_matching_points,
        f"{base_path}/{case_name}/shape/matching/raw_matching.png",
    )

    # Do PnP optimization
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case {case_name}.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw = np.eye(4, dtype=np.float32)
    mesh2raw[:3, :3] = rotation_matrix
    mesh2raw[:3, 3] = tvec.squeeze()

    pnp_camera_pose = np.eye(4, dtype=np.float32)
    pnp_camera_pose[:3, :3] = np.linalg.inv(rotation_matrix)
    pnp_camera_pose[3, :3] = tvec.squeeze()  # change due to pytorch3D setting
    pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]  # change due to pytorch3D setting
    color, depth = render_image(
        mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
    )
    vis_mask = depth > 0
    color[0][~vis_mask] = raw_img[~vis_mask]
    plt.imsave(f"{base_path}/{case_name}/shape/matching/pnp_results.png", color[0])

    
