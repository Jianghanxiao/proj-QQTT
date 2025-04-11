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


if __name__ == "__main__":
    existDir(output_dir)

    cam_idx = 0
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    pcd_path = f"{base_path}/{case_name}/shape_prior.ply"
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
    # Do the alpha shape
    alpha = 0.03
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
    )
    # mesh.show()

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

    alpha = 0.03
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(obs_pcd, alpha)
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    o3d.visualization.draw_geometries([mesh_o3d, obs_pcd, coordinate])


    
    # new_mesh = trimesh.Trimesh(
    #     vertices=np.asarray(new_mesh_o3d.vertices),
    #     faces=np.asarray(new_mesh_o3d.triangles),
    # )


    # # Scale and do the ICP again
    # obs_dim = np.max(obs_points, axis=0) - np.min(obs_points, axis=0)
    # new_dim = np.max(new_pcd.points, axis=0) - np.min(new_pcd.points, axis=0)
    # optimal_scale = obs_dim / new_dim
    # obs_center = np.mean(obs_points, axis=0)
    # new_center = np.mean(new_pcd.points, axis=0)

    # new_points = np.array(new_pcd.points)
    # new_points = (new_points - new_center) * optimal_scale + obs_center
    # new_pcd.points = o3d.utility.Vector3dVector(new_points)
    

    # new_pcd = new_pcd.scale(optimal_scale, center=new_pcd.get_center())
    # new_pcd.translate(-(new_center - obs_center), relative=False)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([new_pcd, obs_pcd, coordinate])

    # Compute the rigid transformation from the original mesh to the final world coordinate
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = np.dot(c2w, np.dot(scale_matrix, mesh2raw_camera))

    mesh_matching_points_world = np.dot(
        mesh2world,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_world = mesh_matching_points_world[:, :3]

    # Do the ARAP based on the matching keypoints
    # Convert the mesh to open3d to use the ARAP function
    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)

    # ARAP based on the keypoints
    deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )

    # Do the ARAP based on both the ray-casting matching and the keypoints
    # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
    final_mesh_world = deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        matching_points,
    )

    if VIS:
        final_mesh_world.compute_vertex_normals()

        # Visualize the partial observation and the mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            f"{output_dir}/final_matching.mp4", fourcc, 30, (width, height)
        )
        # final_mesh_world.compute_vertex_normals()
        # final_mesh_world.translate([0, 0, 0.2])
        # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
        # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
        vis.add_geometry(pcd)
        vis.add_geometry(final_mesh_world)
        # vis.add_geometry(coordinate)
        view_control = vis.get_view_control()

        for j in range(360):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        vis.destroy_window()

    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
    mesh.export(f"{output_dir}/final_mesh.glb")
