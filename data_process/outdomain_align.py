import cv2
import json
import numpy as np
import os
import trimesh
import open3d as o3d
from utils.align_util import plot_image_with_points, select_point, as_mesh
import pickle
import matplotlib.pyplot as plt
from match_pairs import image_pair_matching
from scipy.spatial import KDTree
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, default="./data/different_types")
parser.add_argument("--exp_path", type=str, default="./experiments")
parser.add_argument("--from_case", type=str, default="rope_double_hand")
parser.add_argument("--to_case", type=str, default="single_push_rope")
args = parser.parse_args()

base_path = args.base_path
exp_path = args.exp_path
from_case = args.from_case
to_case = args.to_case

OUTPUT_DIR = f"experiments_out_domain/{args.from_case}_to_{args.to_case}"

CONTROLLER_NAME = "hand"
VIS = True


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_crop_image(img, mask_img):
    # Get the bounding box of the object
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
        min(img.shape[1], bbox[2]),
        min(img.shape[0], bbox[3]),
    )
    # Get the masked cropped image used for superglue
    crop_img = img.copy()
    mask_bool = mask_img > 0
    crop_img[~mask_bool] = 0
    crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

    return crop_img, bbox


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
    # Solve the PNP and verify the reprojection error
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    assert success, "solvePnP failed"
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
        print(f"solvePnP failed for this case.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    return mesh2raw_camera


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


if __name__ == "__main__":
    # Get the two first-frame RGB images to do the SuperGlue matching
    from_image_path = f"{base_path}/{from_case}/color/0/0.png"
    to_image_path = f"{base_path}/{to_case}/color/0/0.png"
    # Get the mask index of the object
    with open(f"{base_path}/{from_case}/mask/mask_info_0.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    from_mask_img_path = f"{base_path}/{from_case}/mask/0/{obj_idx}/0.png"
    # Get the mask index of the object
    with open(f"{base_path}/{to_case}/mask/mask_info_0.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    to_mask_img_path = f"{base_path}/{to_case}/mask/0/{obj_idx}/0.png"

    # Get the cropped mask image from the from_image
    from_img = cv2.imread(from_image_path)
    from_img = cv2.cvtColor(from_img, cv2.COLOR_BGR2RGB)
    from_mask_img = cv2.imread(from_mask_img_path, cv2.IMREAD_GRAYSCALE)
    from_crop_img, from_bbox = get_crop_image(from_img, from_mask_img)
    # Get the cropped mask image from the to_image
    to_img = cv2.imread(to_image_path)
    to_img = cv2.cvtColor(to_img, cv2.COLOR_BGR2RGB)
    to_mask_img = cv2.imread(to_mask_img_path, cv2.IMREAD_GRAYSCALE)
    to_crop_img, to_bbox = get_crop_image(to_img, to_mask_img)

    # Use superglue to match the features
    _, match_result = image_pair_matching(
        [from_crop_img],
        to_crop_img,
        OUTPUT_DIR,
        viz_best=True,
        keypoint_threshold=0.01,
        match_threshold=0.3,
    )

    # Process the matching points
    valid_matches = match_result["matches"] > -1
    from_matching_points_box = match_result["keypoints0"][valid_matches]
    from_matching_points = from_matching_points_box + np.array(
        [from_bbox[0], from_bbox[1]]
    )
    to_matching_points = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    to_matching_points = to_matching_points + np.array([to_bbox[0], to_bbox[1]])

    # Get the 3D points form the from_matching_points
    # Load the pcd in world coordinate of raw image matching points
    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{from_case}/pcd/0.npz"
    mask_path = f"{base_path}/{from_case}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    from_points = data["points"][0]
    from_colors = data["colors"][0]
    from_mask = processed_masks[0][0]["object"]

    # Find the cloest points for the raw_matching_points
    from_new_match, from_matching_points_3d = select_point(
        from_points, from_matching_points, from_mask
    )

    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{to_case}/pcd/0.npz"
    mask_path = f"{base_path}/{to_case}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    to_points = data["points"][0]
    to_colors = data["colors"][0]
    to_mask = processed_masks[0][0]["object"]

    to_new_match, to_matching_points_3d = select_point(
        to_points, to_matching_points, to_mask
    )

    if VIS:
        # Draw the raw_matching_points and new matching points on the masked
        vis_img = from_img.copy()
        vis_img[~from_mask] = 0
        plot_image_with_points(
            vis_img,
            from_matching_points,
            f"{OUTPUT_DIR}/from_matching_valid.png",
            from_new_match,
        )

    with open(f"{base_path}/{to_case}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsic = np.array(data["intrinsics"])[0]

    with open(f"{base_path}/{to_case}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
        c2w = c2ws[0]

    # Directly do the PNP between the from_matching_points_3d and to_matching_points (2D)
    from_to_to_camera = registration_pnp(
        from_matching_points_3d, to_matching_points, intrinsic
    )
    from_to_to = np.dot(c2w, from_to_to_camera)

    # Load the final from object points
    data_path = f"{base_path}/{from_case}/final_data.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    from_object_points = data["object_points"]
    from_object_colors = data["object_colors"]
    from_other_surface_points = data["surface_points"]
    from_interior_points = data["interior_points"]
    from_structure_points = np.concatenate(
        [from_object_points[0], from_other_surface_points, from_interior_points], axis=0
    )
    # Load the final to object points
    data_path = f"{base_path}/{to_case}/final_data.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    to_object_points = data["object_points"]
    to_object_colors = data["object_colors"]
    to_other_surface_points = data["surface_points"]
    to_interior_points = data["interior_points"]
    to_structure_points = np.concatenate(
        [to_object_points[0], to_other_surface_points, to_interior_points], axis=0
    )

    from_structure_points_to = np.dot(
        from_to_to,
        np.hstack(
            [from_structure_points, np.ones((from_structure_points.shape[0], 1))]
        ).T,
    ).T[:, :3]

    vis_to_object_colors = np.concatenate(
        [
            to_object_colors[0],
            np.ones(
                (
                    to_structure_points.shape[0] - to_object_colors.shape[1],
                    3,
                )
            )
            * 0.3,
        ],
        axis=0,
    )

    to_pcd = o3d.geometry.PointCloud()
    to_pcd.points = o3d.utility.Vector3dVector(to_structure_points)
    to_pcd.colors = o3d.utility.Vector3dVector(vis_to_object_colors)

    vis_from_object_colors = np.concatenate(
        [
            from_object_colors[0],
            np.ones(
                (
                    from_structure_points.shape[0] - from_object_colors.shape[1],
                    3,
                )
            )
            * 0.3,
        ],
        axis=0,
    )

    from_pcd_to = o3d.geometry.PointCloud()
    from_pcd_to.points = o3d.utility.Vector3dVector(from_structure_points_to)
    from_pcd_to.colors = o3d.utility.Vector3dVector(vis_from_object_colors)

    if "zebra" in from_case or "sloth" in from_case:
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]
        current_transformation = np.identity(4)
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]

            source_down = from_pcd_to.voxel_down_sample(radius)
            target_down = to_pcd.voxel_down_sample(radius)

            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down,
                target_down,
                radius,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                ),
            )
            current_transformation = result_icp.transformation
    else:
        # Do ICP to align the two point clouds
        threshold = 0.02
        init_transform = np.eye(4)
        init_transform[:3, 3] = np.mean(np.asarray(to_pcd.points), axis=0) - np.mean(
            np.asarray(from_pcd_to.points), axis=0
        )
        reg_p2p = o3d.pipelines.registration.registration_icp(
            from_pcd_to,
            to_pcd,
            threshold,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        current_transformation = reg_p2p.transformation

    from_pcd_to.transform(current_transformation)
    from_matching_points_to = np.dot(
        np.dot(current_transformation, from_to_to),
        np.hstack(
            [
                from_matching_points_3d,
                np.ones((from_matching_points_3d.shape[0], 1)),
            ]
        ).T,
    ).T[:, :3]

    final_points = np.asarray(from_pcd_to.points)
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_points)
    final_pcd.paint_uniform_color([0, 0, 1])

    # Do the ARAP with the shape prior
    if "cloth" not in from_case:
        # Reset the matching points to the original points if they are very close
        matching_points_flag = (
            np.linalg.norm(from_matching_points_to - to_matching_points_3d, axis=1)
            < 0.02
        )

        to_matching_points_3d[matching_points_flag] = from_matching_points_to[
            matching_points_flag
        ]

        # Create the KDTree for the from_matching_points_to
        kdtree = KDTree(from_matching_points_to)
        _, indices = kdtree.query(to_matching_points_3d, k=3)
        additional_flag = np.zeros_like(matching_points_flag)
        for i in range(to_matching_points_3d.shape[0]):
            if not matching_points_flag[i]:
                matching_num = 0
                for j in range(3):
                    if matching_points_flag[indices[i, j]]:
                        matching_num += 1
                if matching_num >= 1:
                    additional_flag[i] = True
        matching_points_flag = np.logical_or(matching_points_flag, additional_flag)

        from_matching_points_to = from_matching_points_to[~additional_flag]
        to_matching_points_3d = to_matching_points_3d[~additional_flag]

        if (~matching_points_flag).sum() != 0:
            # Load the shape prior
            mesh_path = f"{base_path}/{from_case}/shape/matching/final_mesh.glb"
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            mesh = as_mesh(mesh)
            initial_mesh = o3d.geometry.TriangleMesh()
            initial_mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices)
            )
            initial_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            initial_mesh = initial_mesh.remove_duplicated_vertices()
            initial_mesh = initial_mesh.remove_duplicated_triangles()
            initial_mesh = initial_mesh.remove_degenerate_triangles()
            initial_mesh.compute_vertex_normals()
            initial_mesh.transform(np.dot(current_transformation, from_to_to))

            deform_mesh, from_mesh_points_indices = deform_ARAP(
                initial_mesh, from_matching_points_to, to_matching_points_3d
            )
            # to_pcd.paint_uniform_color([1, 0, 0])
            # deform_mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([deform_mesh, from_pcd_to, to_pcd])

            # Do the interpolation for all points
            original_vertices = np.asarray(initial_mesh.vertices)
            deform_vertices = np.asarray(deform_mesh.vertices)
            motions = deform_vertices - original_vertices

            kdtree = KDTree(original_vertices)
            _, indices = kdtree.query(final_points, k=5)
            weights = np.zeros((final_points.shape[0], 5))
            for i in range(final_points.shape[0]):
                for j in range(5):
                    weights[i, j] = 1 / np.linalg.norm(
                        final_points[i] - original_vertices[indices[i, j]]
                    )
                weights[i] = weights[i] / weights[i].sum()

            final_points = np.zeros_like(final_points)
            for i in range(final_points.shape[0]):
                final_points[i] = (
                    original_vertices[indices[i]] + motions[indices[i]]
                ).T @ weights[i]

            index = np.where(final_points[:, 2] > 0)[0]
            final_points[index, 2] = 0

            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(final_points)
            final_pcd.paint_uniform_color([0, 0, 1])

    if VIS:
        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            f"{OUTPUT_DIR}/rigid_matching.mp4", fourcc, 30, (width, height)
        )

        # Visualzie the matching points
        vis_from_matching_pcd = o3d.geometry.PointCloud()
        vis_from_matching_pcd.points = o3d.utility.Vector3dVector(
            from_matching_points_to
        )
        vis_from_matching_pcd.paint_uniform_color([1, 0, 0])

        vis_to_matching_pcd = o3d.geometry.PointCloud()
        vis_to_matching_pcd.points = o3d.utility.Vector3dVector(to_matching_points_3d)
        vis_to_matching_pcd.paint_uniform_color([0, 1, 0])

        to_pcd.paint_uniform_color([1, 0.7, 0.7])

        vis.add_geometry(from_pcd_to)
        vis.add_geometry(vis_from_matching_pcd)
        vis.add_geometry(vis_to_matching_pcd)
        vis.add_geometry(to_pcd)
        vis.add_geometry(final_pcd)
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

        video_writer.release()

    # Save the final points position
    with open(f"{OUTPUT_DIR}/final_points.pkl", "wb") as f:
        pickle.dump(final_points, f)