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
    select_point,
)
from match_pairs import image_pair_matching
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree

VIS = True

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
output_dir = f"{base_path}/{case_name}/shape/new_matching"


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
    existDir(output_dir)

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

    # Load the c2w for the camera
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
        c2w = c2ws[cam_idx]
        w2c = np.linalg.inv(c2w)

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

        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
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

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_dir}/mesh_matching.png",
        )
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_dir}/render_matching.png",
        )
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_dir}/raw_matching.png",
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
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)
        pnp_camera_pose[:3, :3] = np.linalg.inv(rotation_matrix)
        pnp_camera_pose[3, :3] = tvec.squeeze()  # change due to pytorch3D setting
        pnp_camera_pose[:, :2] = -pnp_camera_pose[
            :, :2
        ]  # change due to pytorch3D setting
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )
        vis_mask = depth > 0
        color[0][~vis_mask] = raw_img[~vis_mask]
        plt.imsave(f"{output_dir}/pnp_results.png", color[0])

    # Transform the mesh into the real world coordinate
    mesh_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_points_cam = mesh_points_cam[:, :3]

    # Load the pcd in world coordinate of raw image matching points
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    points = data["points"][cam_idx]
    colors = data["colors"][cam_idx]
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    object_mask = processed_masks[cam_idx][0]["object"]

    # Find the cloest points for the raw_matching_points
    new_match, matched_points = select_point(points, raw_matching_points, object_mask)
    matched_points_cam = np.dot(
        w2c, np.hstack((matched_points, np.ones((matched_points.shape[0], 1)))).T
    ).T
    matched_points_cam = matched_points_cam[:, :3]

    if VIS:
        # Draw the raw_matching_points on the masked
        vis_img = raw_img.copy()
        vis_img[~object_mask] = 0
        plot_image_with_points(
            vis_img,
            raw_matching_points,
            f"{output_dir}/raw_matching_valid.png",
            new_match,
        )

    def objective(scale, mesh_points, pcd_points):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 1
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_points_cam, matched_points_cam),
        method="L-BFGS-B",
    )
    optimal_scale = result.x[0]
    print("Rescale:", optimal_scale)

    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    scale_camera = np.dot(scale_matrix, mesh2raw_camera)
    final_transform = np.dot(c2w, scale_camera)

    mesh_points_world = np.dot(
        final_transform,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_points_world = mesh_points_world[:, :3]

    # Do the ARAP deformation based on the points
    cur_mesh = o3d.io.read_triangle_mesh(mesh_path)
    cur_mesh.transform(final_transform)
    cur_mesh = cur_mesh.compute_triangle_normals()
    cur_mesh = cur_mesh.remove_duplicated_vertices()
    cur_mesh = cur_mesh.remove_duplicated_triangles()
    cur_mesh = cur_mesh.remove_unreferenced_vertices()
    cur_mesh = cur_mesh.filter_smooth_simple()
    # Get the vertex indices of the mesh_points
    mesh_vertices = np.asarray(cur_mesh.vertices)  # 获取 mesh 顶点
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    # import pdb

    # pdb.set_trace()
    new_mesh = cur_mesh.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matched_points),
        max_iter=1,
    )
    new_mesh.compute_vertex_normals()
    new_mesh.paint_uniform_color([0, 0.5, 0])

    # Visualize in 3D PCD and the mesh
    # points_cam = np.dot(
    #     w2c, np.hstack((points[object_mask], np.ones((points[object_mask].shape[0], 1)))).T
    # ).T
    # points_cam = points_cam[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[object_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[object_mask] / 255)

    vis_mesh = o3d.io.read_triangle_mesh(mesh_path)
    vis_mesh.compute_vertex_normals()
    vis_mesh.transform(final_transform)
    vis_mesh.paint_uniform_color([0.5, 0, 0])

    mesh_keypoint = o3d.geometry.PointCloud()
    mesh_keypoint.points = o3d.utility.Vector3dVector(mesh_points_world)
    mesh_keypoint.paint_uniform_color([1, 0, 0])

    raw_keypoint = o3d.geometry.PointCloud()
    raw_keypoint.points = o3d.utility.Vector3dVector(matched_points)
    raw_keypoint.paint_uniform_color([0, 0, 1])

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries(
    #     [pcd, vis_mesh, mesh_keypoint, raw_keypoint, coordinate]
    # )
    o3d.visualization.draw_geometries([pcd, new_mesh, raw_keypoint, coordinate])

    # # combine pose and transformation
    # S = np.array([[optimal_scale, 0, 0, 0],
    #               [0, optimal_scale, 0, 0],
    #               [0, 0, optimal_scale, 0],
    #               [0, 0, 0, 1]])
    # M = np.dot(S, world_2_cam)
    # print('final matrix')
    # print(np.array2string(M, separator=', '))
    # print('plane model')
    # print(np.array2string(plane_model, separator=', '))
    # print('field of view')
    # print(np.array2string(field_of_view, separator=', '))

    # import pdb

    # pdb.set_trace()
