import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import pickle
import trimesh
import cv2
import json
import torch
import os
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    AmbientLights,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from match_pairs import image_pair_matching
import matplotlib.pyplot as plt

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
parser = ArgumentParser()
parser.add_argument("--case_name", type=str, default="double_stretch_zebra")
args = parser.parse_args()
case_name = args.case_name
print(f"Processing {case_name}")
reference_path = "/home/hanxiao/Desktop/Research/proj-qqtt/TRELLIS/outputs_sdxl/outputs_zebra/object.glb"
CONTROLLER_NAME = "hand"


def sample_camera_poses(radius, num_samples, num_up_samples=4, device="cpu"):
    """
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    """
    camera_poses = []
    phi = np.linspace(0, np.pi, num_samples)  # Elevation angle
    phi = phi[1:-1]  # Exclude poles
    theta = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angle

    # Generate different up vectors
    up_vectors = [np.array([0, 0, 1])]  # z-axis is up
    for i in range(1, num_up_samples):
        angle = (i / num_up_samples) * np.pi * 2
        up = np.array([np.sin(angle), 0, np.cos(angle)])  # Rotate around y-axis
        up_vectors.append(up)

    for p in phi:
        for t in theta:
            for up in up_vectors:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                position = np.array([x, y, z])[None, :]
                lookat = np.array([0, 0, 0])[None, :]
                up = up[None, :]
                R, T = look_at_view_transform(radius, t, p, False, position, lookat, up)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R
                camera_pose[3, :3] = T
                camera_poses.append(camera_pose)

    print("total poses", len(camera_poses))
    return torch.tensor(np.array(camera_poses), device=device)


def render_image(mesh, camera_poses, width=640, height=480, fov=1, device="cpu"):
    camera_poses = torch.tensor(camera_poses, device=device)
    if len(camera_poses.shape) == 2:
        camera_poses = camera_poses[None, :]

    from pytorch3d.io import IO
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(mesh)
    mesh = mesh.to(device)

    R = camera_poses[:, :3, :3]
    T = camera_poses[:, 3, :3]
    num_poses = camera_poses.shape[0]
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        device=device,
        focal_length=torch.ones(num_poses, 1)
        * 0.5
        * width
        / np.tan(fov / 2),  # Calculate focal length from FOV in radians
        principal_point=torch.tensor((width / 2, height / 2))
        .repeat(num_poses)
        .reshape(-1, 2),  # different order from image_size!!
        image_size=torch.tensor((height, width)).repeat(num_poses).reshape(-1, 2),
        in_ndc=False,
    )

    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            blend_params=BlendParams(background_color=(0, 0, 0)),
            cameras=cameras,
            lights=lights,
        ),
    )
    extended_mesh = mesh.extend(num_poses).to(device)
    fragments = renderer.rasterizer(extended_mesh)
    depth = fragments.zbuf.squeeze().cpu().numpy()
    rendered_images = renderer(mesh.extend(num_poses))
    color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)

    return color, depth


def render_multi_images(
    mesh,
    width=640,
    height=480,
    fov=1,
    radius=3.0,
    num_samples=6,
    num_ups=2,
    device="cpu",
):
    # Sample camera poses
    camera_poses = sample_camera_poses(radius, num_samples, num_ups, device)

    # Calculate intrinsics
    fx = 0.5 * width / np.tan(fov / 2)
    fy = fx  # * aspect_ratio
    cx, cy = width / 2, height / 2
    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    num_cameras = camera_poses.shape[0]

    # Render two times to avoid memory overflow
    split = num_cameras // 2
    color1, depth1 = render_image(
        mesh, camera_poses[:split], width, height, fov, device
    )
    color2, depth2 = render_image(
        mesh, camera_poses[split:], width, height, fov, device
    )
    color = np.concatenate([color1, color2], axis=0)
    depth = np.concatenate([depth1, depth2], axis=0)
    return color, depth, camera_poses, camera_intrinsics

def plot_mesh_with_points(mesh, points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, alpha=0.5, edgecolor='none', color='lightgrey')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title('3D Mesh with Projected Points')
    plt.savefig(filename)
    plt.clf()

def plot_image_with_points(image, points, save_dir):
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], color='red', s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points on Original Image')
    plt.savefig(save_dir)
    plt.clf()

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):

        # Extract all meshes from the scene
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        # Combine all meshes if there are multiple
        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        elif len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            raise ValueError("No valid meshes found in the GLB file")
        
        # Get model metadata
        metadata = {
            'vertices': combined_mesh.vertices.shape[0],
            'faces': combined_mesh.faces.shape[0],
            'bounds': combined_mesh.bounds.tolist(),
            'center_mass': combined_mesh.center_mass.tolist(),
            'is_watertight': combined_mesh.is_watertight,
            'original_scene': combined_mesh  # Keep reference to original scene
        }

        mesh = combined_mesh
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def get_bbox(img):
    """Get bounding box of non-zero pixels in the image."""
    non_zero_coords = np.where(img != 0)
    ymin, ymax = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
    xmin, xmax = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
    return xmin, ymin, xmax, ymax

def project_2d_to_3d(image_points, depth, camera_intrinsics, camera_pose):
    """
    Project 2D image points to 3D space using the depth map, camera intrinsics, and pose.

    :param image_points: Nx2 array of image points
    :param depth: Depth map
    :param camera_intrinsics: Camera intrinsic matrix
    :param camera_pose: 4x4 camera pose matrix
    :return: Nx3 array of 3D points in world coordinates
    """
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    # Convert image points to normalized device coordinates (NDC)
    ndc_points = np.zeros((image_points.shape[0], 3))
    for i, (u, v) in enumerate(image_points):
        z = depth[int(v), int(u)]
        x = - (u - cx) * z / fx
        y = - (v - cy) * z / fy
        ndc_points[i] = [x, y, z]
    valid_mask = ndc_points[:, 2] > 0
    ndc_points = ndc_points[valid_mask]
    # ndc_points = np.vstack((ndc_points, np.zeros(3), [[0, 0, 0]])) # modified
    # Convert from camera coordinates to world coordinates
    ndc_points_homogeneous = np.hstack((ndc_points, np.ones((ndc_points.shape[0], 1))))
    world_points_homogeneous = ndc_points_homogeneous @ np.linalg.inv(camera_pose)
    return world_points_homogeneous[:, :3], valid_mask


if __name__ == "__main__":
    cam_idx = 0
    raw_img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    depth_path = f"{base_path}/{case_name}/depth/{cam_idx}/0.npy"
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
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))
    w2c = np.linalg.inv(c2ws[cam_idx])

    # Load the complete reference mesh
    mesh = trimesh.load_mesh(reference_path, force="mesh")
    mesh = as_mesh(mesh)

    # Load and process the original image
    raw_img = cv2.imread(raw_img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # Get mask bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    mask_box = get_bbox(mask_img)
    print("Mask bounding box:", mask_box)

    # Create reference image from mask
    ref_img = np.zeros_like(np.array(raw_img))
    mask_bool = mask_img > 0
    ref_img[mask_bool] = raw_img[mask_bool]
    ref_img = ref_img[mask_box[1] : mask_box[3], mask_box[0] : mask_box[2]]
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)
    print("rendering radius:", radius)

    # Render multimle images and feature matching
    print("rendering objects...")
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        reference_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]

    print("matching features...")
    os.system("rm -rf ./test_output")

    best_pose, match_result = image_pair_matching(
        grays, ref_img, "./test_output", viz=True
    )
    chosen_pose = camera_poses[best_pose].cpu().numpy()
    print("best_pose", np.array2string(chosen_pose, separator=", "))
    print("matched point number", np.sum(match_result["matches"] > -1))
    # cv2.imshow("Reference image", colors[best_pose])
    # cv2.waitKey(0)

    valid_matches = match_result['matches'] > -1
    image_points = match_result['keypoints0'][valid_matches]
    world_points, valid_mask = project_2d_to_3d(
        image_points, 
        depths[best_pose],
        camera_intrinsics, 
        chosen_pose
    )
    image_points = image_points[valid_mask]

    # Visualize the correpsonded points on the mesh and on the image
    plot_mesh_with_points(mesh, world_points, "test1.png")
    plot_image_with_points(depths[best_pose], image_points, "test2.png")

    # Process matched points on original raw frame
    # - keypoints1 is of the original image
    # - match_points_on_raw: 2D points of keypoints1 in pixel coordinates
    match_points_on_mask = match_result['keypoints1'][match_result['matches'][valid_matches]]
    match_points_on_mask = match_points_on_mask[valid_mask]
    scale_x = (mask_box[2]-mask_box[0]) / ref_img.shape[1]
    scale_y = (mask_box[3]-mask_box[1]) / ref_img.shape[0]
    match_points_on_raw = match_points_on_mask * np.array([scale_x, scale_y]) + np.array([mask_box[0], mask_box[1]])
    plot_image_with_points(raw_img, match_points_on_raw, "test_3.png")

    


