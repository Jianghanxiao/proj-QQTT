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
output_dir = f"{base_path}/{case_name}/shape/matching"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
    # Do the ARAP deformation based on the matching keypoints
    mesh_vertices = np.asarray(initial_mesh_world.vertices)
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )
    return deform_mesh, mesh_points_indices


def get_matching_ray_registration(
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
    # Get the matching indices and targets based on the viewpoint
    obs_points_cam = np.dot(
        w2c,
        np.hstack((obs_points_world, np.ones((obs_points_world.shape[0], 1)))).T,
    ).T
    obs_points_cam = obs_points_cam[:, :3]
    vertices_cam = np.dot(
        w2c,
        np.hstack(
            (
                np.asarray(mesh_world.vertices),
                np.ones((np.asarray(mesh_world.vertices).shape[0], 1)),
            )
        ).T,
    ).T
    vertices_cam = vertices_cam[:, :3]

    obs_kd = KDTree(obs_points_cam)

    new_indices = []
    new_targets = []
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    for index, vertex in enumerate(vertices_cam):
        ray_origins = np.array([[0, 0, 0]])
        ray_direction = vertex
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        ray_directions = np.array([ray_direction])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )

        ignore_flag = False

        if len(locations) > 0:
            first_intersection = locations[0]
            vertex_distance = np.linalg.norm(vertex)
            intersection_distance = np.linalg.norm(first_intersection)
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True

        if ignore_flag:
            continue
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)
            line_distances = line_point_distance(vertex, obs_points_cam[indices])
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )
                new_indices.append(index)
                new_targets.append(target[:3])

    new_indices = np.asarray(new_indices)
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets


def deform_ARAP_ray_registration(
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
    final_indices = []
    final_targets = []
    for index, target in zip(mesh_points_indices, matching_points):
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )
    return final_mesh_world


def line_point_distance(p, points):
    # Compute the distance between points and the line between p and [0, 0, 0]
    p = p / np.linalg.norm(p)
    points_to_origin = points
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)
    return cross_product / np.linalg.norm(p)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.5
    print(
        ":: Apply fast global registration with distance threshold %.3f"
        % distance_threshold
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


if __name__ == "__main__":
    existDir(output_dir)

    cam_idx = 0
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    pcd_path = f"{base_path}/{case_name}/scan.pcd"
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
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]

    # Load the shape prior
    pcd = o3d.io.read_point_cloud(pcd_path)
    intial_rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    pcd = pcd.rotate(intial_rotation, center=(0, 0, 0))

    # Load the pcd in world coordinate of raw image matching points
    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    for i in range(3):
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])
        if i == 0:
            first_points = points
            first_mask = mask

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_points)
    obs_pcd.colors = o3d.utility.Vector3dVector(obs_colors)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # pcd.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pcd, obs_pcd, coordinate])

    voxel_size = 0.02
    source_down, source_fpfh = preprocess_point_cloud(pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(obs_pcd, voxel_size)
    result_ransac = execute_fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    pcd = pcd.transform(result_ransac.transformation)
    # pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd, obs_pcd, coordinate])

    # Do the ICP
    threshold = 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        obs_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    align_pcd = pcd.transform(reg_p2p.transformation)
    # align_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([align_pcd, obs_pcd, coordinate])

    # Do the alpha shape
    alpha = 0.03
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        align_pcd, alpha
    )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        vertex_colors=(np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8),
    )
    # mesh.show()

    mesh.export(f"{output_dir}/final_mesh.glb")
