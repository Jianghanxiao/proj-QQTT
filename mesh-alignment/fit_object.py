import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
import sys
import json
import torch
import pickle
from pytorch3d.renderer import (
    look_at_view_transform, PerspectiveCameras, PointLights, RasterizationSettings,
    BlendParams, MeshRenderer, MeshRasterizer, SoftPhongShader
)
from pytorch3d.ops import iterative_closest_point
from pytorch3d.transforms import Transform3d
from kornia import create_meshgrid
import open3d as o3d

# Set up environment variables
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Append the current directory to the system path
sys.path.append(os.getcwd())
from match_pairs import image_pair_matching


def sample_camera_poses(radius, num_samples, num_up_samples=4, device='cpu'):
    '''
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    '''
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

                # # Attention! Negative! Not normal forward!
                # forward = -(lookat - position) / np.linalg.norm(lookat - position)
                # right = np.cross(up, forward)
                # if np.linalg.norm(right) < 1e-6:  # Check for collinearity
                #     right = np.array([1, 0, 0])
                # right = right / np.linalg.norm(right)
                # up = np.cross(forward, right)

                # camera_pose = np.eye(4)
                # camera_pose[:3, 0] = right
                # camera_pose[:3, 1] = up
                # camera_pose[:3, 2] = forward
                # camera_pose[:3, 3] = position
                # print('old', camera_pose[:3, :3], position)
    print('total poses', len(camera_poses))
    return torch.tensor(np.array(camera_poses), device=device)


def render_image(mesh, camera_poses, width=640, height=480, fov=1, device='cpu'):
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
    cameras = PerspectiveCameras(R=R, T=T, device=device,
        focal_length=torch.ones(num_poses, 1) * 0.5 * width / np.tan(fov / 2),  # Calculate focal length from FOV in radians
        principal_point=torch.tensor((width/2, height/2)).repeat(num_poses).reshape(-1, 2),  #different order from image_size!!
        image_size=torch.tensor((height, width)).repeat(num_poses).reshape(-1, 2),
        in_ndc=False)
    # print(cameras.get_world_to_view_transform().get_matrix()) # T is matrix[3, :3]
    light_location = torch.linalg.inv(camera_poses)[:, 3, :3]
    lights = PointLights(location=light_location, device=device)
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
            blend_params=BlendParams(background_color=(0,0,0)),
            cameras=cameras,
            lights=lights
        )
    )
    extended_mesh = mesh.extend(num_poses).to(device)
    fragments = renderer.rasterizer(extended_mesh)
    depth = fragments.zbuf.squeeze().cpu().numpy()
    rendered_images = renderer(mesh.extend(num_poses))
    color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)
    # if num_poses>1:
    #     for i in range(num_poses):
    #         plt.imsave(f'locate/render_result/rendered_image_{i}.png', color[i])
    #         plt.imsave(f'locate/render_result/depth_image_{i}.png', depth[i])
    return color, depth


def render_multi_images(mesh, width=640, height=480, fov=1, radius=3.0, num_samples=6, num_ups=2, device='cpu'):
    # Sample camera poses
    camera_poses = sample_camera_poses(radius, num_samples, num_ups, device)

    # Calculate intrinsics
    # aspect_ratio = width / height # modified
    fx = 0.5 * width / np.tan(fov / 2)
    fy = fx # * aspect_ratio
    cx, cy = width / 2, height / 2
    camera_intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    color, depth = render_image(mesh, camera_poses, width, height, fov, device)
    return color, depth, camera_poses, camera_intrinsics


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


def plot_mesh_with_points(mesh, points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, alpha=0.5, edgecolor='none', color='lightgrey')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=10)
    # ax.scatter([2.853], [0], [0.927], color='green', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title('3D Mesh with Projected Points')
    # for i in range(10):
    #     angle = 360 / 5 * i
    #     ax.view_init(elev=10., azim=angle)
    #     plt.savefig(filename.split('.')[0] + f'_{i}.png')
    plt.savefig(filename)
    plt.clf()
    # new_mesh = mesh.copy()
    # new_vertices = np.vstack((new_mesh.vertices, points))
    # new_faces = new_mesh.faces.copy()
    # new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    # new_mesh.export(filename.split('.')[0] + '.ply', file_type='ply')


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


def convert_coord_world_2_cam(points, camera_pose):
    '''
    Convert 3D points from world coordinates to camera coordinates.

    points: Nx3 array of 3D points in world coordinates
    camera_pose: 4x4 world-to-camera transformation matrix

    return: Nx3 array of 3D points in camera coordinates
    '''
    # Convert world coordinates to camera coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    camera_points = points_h @ camera_pose.T
    camera_points = camera_points[:, :3]
    return camera_points


def get_bbox(img):
    """Get bounding box of non-zero pixels in the image."""
    non_zero_coords = np.where(img != 0)
    ymin, ymax = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
    xmin, xmax = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
    return xmin, ymin, xmax, ymax


def estimate_pose(
    mesh_file: str,
    pcd_file: str,
    raw_img_path: str,
    mask_img: np.ndarray,
    mask_box: tuple,
    out_dir: str,
    K: np.ndarray,
    pose: np.ndarray,
    depth_path: str,
    num_samples: int = 8,
    num_ups: int = 1
) -> None:
    """
    Estimate camera pose and scale the mesh.
    
    Args:
        mesh_file (str): Path to the reconstructed mesh file
        pcd_file (str): Path to the unprojected point cloud file
        raw_img_path (str): Path to the original color image
        mask_img (np.ndarray): Mask image to filter the raw image
        mask_box (tuple): Bounding box of the mask in the raw image (x1, y1, x2, y2)
        out_dir (str): Directory to save output files
        K (np.ndarray): Camera intrinsic matrix
        pose (np.ndarray): Camera pose (world-to-camera transformation matrix)
        depth_path (str): Path to the depth map file
        num_samples (int): Number of camera poses to sample
        num_ups (int): Number of up vectors to use for camera pose sampling
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and process mesh
    mesh = trimesh.load_mesh(mesh_file, force='mesh')
    mesh = as_mesh(mesh)

    # Load and process original image
    raw_img = cv2.imread(raw_img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # Create reference image from mask
    ref_img = np.zeros_like(np.array(raw_img))
    mask_bool = mask_img > 0
    ref_img[mask_bool] = raw_img[mask_bool]
    ref_img = ref_img[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]]   # Crop the mask region
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    
    # Load point cloud data
    pcd_info = np.load(pcd_file)
    xyz_world = pcd_info['points']
    rgb = pcd_info['colors']
    xyz_cam = convert_coord_world_2_cam(xyz_world, pose)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_cam)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * K[0, 0]))

    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)
    print('rendering radius:', radius)

    # Render multimle images and feature matching
    print('rendering objects...')
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(mesh_file, 
                                                                          raw_img.shape[1],
                                                                          raw_img.shape[0], fov, radius=radius,
                                                                          num_samples=num_samples, num_ups=num_ups, device=device)
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]

    print('matching features...')
    best_pose, match_result = image_pair_matching(grays, ref_img, out_dir,
                                    resize=[-1], viz=False, save=False, keypoint_threshold=0.001, match_threshold=0.01)
    chosen_pose = camera_poses[best_pose].cpu().numpy()
    print('best_pose', np.array2string(chosen_pose, separator=', '))
    print('matched point number', np.sum(match_result['matches']>-1))
    plt.imsave(os.path.join(out_dir, 'best_pose_rendering.png'), colors[best_pose])
    # chosen_pose[:, 1:3] = -chosen_pose[:, 1:3] # Change due to pyrender's special coordinates

    # Process matched points on mesh
    # - keypoints0 is of the rendered mesh image
    # - world_points: 3D points of keypoints0 in world coordinates
    # - image_points: 2D points of keypoints0 in pixel coordinates
    valid_matches = match_result['matches'] > -1
    image_points = match_result['keypoints0'][valid_matches]
    world_points, valid_mask = project_2d_to_3d(
        image_points, 
        depths[best_pose],
        camera_intrinsics, 
        chosen_pose
    )
    image_points = image_points[valid_mask]
    plot_mesh_with_points(mesh, world_points, os.path.join(out_dir, 'points_on_3D.png'))
    plot_image_with_points(depths[best_pose], image_points, os.path.join(out_dir, 'points_on_2D.png'))

    # Process matched points on original raw frame
    # - keypoints1 is of the original image
    # - match_points_on_raw: 2D points of keypoints1 in pixel coordinates
    match_points_on_mask = match_result['keypoints1'][match_result['matches'][valid_matches]]
    match_points_on_mask = match_points_on_mask[valid_mask]
    scale_x = (mask_box[2]-mask_box[0]) / ref_img.shape[1]
    scale_y = (mask_box[3]-mask_box[1]) / ref_img.shape[0]
    match_points_on_raw = match_points_on_mask * np.array([scale_x, scale_y]) + np.array([mask_box[0], mask_box[1]])
    plot_image_with_points(raw_img, match_points_on_raw, os.path.join(out_dir, 'points_original.png'))

    # Process depth information
    depth = np.load(depth_path) / 1000.0  # Convert to meters
    depth_mask = np.logical_and(depth > 0.05, depth < 2.0)
    H, W = depth.shape

    # Unproject to obtain camera points from depth
    # - projection: 2D positions of the mask in pixel coordinates
    grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
    u, v = grid.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], dim=-1).cpu().numpy()
    points_cam = directions * depth[..., None]
    final_mask = np.logical_and(depth_mask, mask_bool)
    points_cam = points_cam[final_mask]
    projection = np.concatenate((u[final_mask].cpu().unsqueeze(-1).numpy(), v[final_mask].cpu().unsqueeze(-1).numpy()), axis=-1)

    # Find closest points of each keypoint1 and convert to world coordinates
    # - world_points_orig: 3D points of keypoints1 in world coordinates
    closest_points = []
    for raw_point in match_points_on_raw:
        distances = np.linalg.norm(projection - raw_point, axis=1)
        closest_points.append(points_cam[np.argmin(distances)])
    closest_points = np.array(closest_points)
    c2w = np.linalg.inv(pose)
    world_points_orig = np.dot(np.hstack((closest_points, np.ones((closest_points.shape[0], 1)))), c2w.T)[:, :3]

    # Get relative scale by majority voting
    N = world_points.shape[0]
    scales = []
    for _ in range(1000):
        i, j = np.random.choice(N, 2, replace=False)
        dist_X = np.linalg.norm(world_points[i] - world_points[j])
        dist_Y = np.linalg.norm(world_points_orig[i] - world_points_orig[j])
        if dist_X > 0:
            scales.append(dist_Y / dist_X)

    # Find majority scale using histogram
    min_scales, max_scales = 0.01, 10
    n_bins = 50
    bins = np.logspace(np.log10(min_scales), np.log10(max_scales), n_bins)
    hist, bin_edges = np.histogram(scales, bins=bins)
    majority_bin_index = np.argmax(hist)
    majority_scale = (bin_edges[majority_bin_index] + bin_edges[majority_bin_index + 1]) / 2
    print("Majority Scale:", majority_scale)

    # Scale the mesh by relative scale
    mesh_scaled = mesh.copy()
    scale_matrix = np.array([
        [majority_scale, 0, 0, 0],
        [0, majority_scale, 0, 0],
        [0, 0, majority_scale, 0],
        [0, 0, 0, 1]
    ])
    mesh_scaled.apply_transform(scale_matrix)

    # run ICP
    source_points = torch.Tensor(mesh_scaled.vertices).to(device)
    target_points = torch.Tensor(xyz_world).to(device)
    source_centroid = source_points.mean(dim=0)
    target_centroid = target_points.mean(dim=0)
    source_points_centered = (source_points - source_centroid)[None]
    target_points_centered = (target_points - target_centroid)[None]

    results = iterative_closest_point(
        X=source_points_centered,
        Y=target_points_centered,
        max_iterations=1000,
        estimate_scale=False,  # set scale=True would mess up
        allow_reflection=False
    )

    # Apply transformation
    R = results.RTs.R  # 1x3x3
    T = results.RTs.T  # 1x3
    s = results.RTs.s  # 1x1
    trans = Transform3d().scale(s).rotate(R).translate(T).to(device)

    # Export final meshes (this part is buggy if CUDA enabled, run with cpu works fine)
    # new_source_points = trans.transform_points(source_points_centered)
    # final_vertices = new_source_points[0].cpu().numpy() + np.array(target_centroid.cpu())
    # source_mesh_data = {
    #     'vertices': final_vertices,
    #     'vertex_colors': (np.ones_like(final_vertices) * 255).astype(np.uint8)  # Ensure colors are in 0-255 range and uint8
    # }
    # source_mesh = trimesh.Trimesh(**source_mesh_data)
    # source_mesh.export(os.path.join(out_dir, 'source.ply'))

    target_mesh_data = {
        'vertices': xyz_world,
        'vertex_colors': (rgb * 255).astype(np.uint8)  # Ensure colors are in 0-255 range and uint8
    }
    target_mesh = trimesh.Trimesh(**target_mesh_data)
    target_mesh.export(os.path.join(out_dir, 'target.ply'))

    # Aggregate all transformations into a single 4x4 matrix
    offset_trans = target_centroid.cpu().numpy()
    offset_mat = np.array([
        [1, 0, 0, offset_trans[0]],
        [0, 1, 0, offset_trans[1]],
        [0, 0, 1, offset_trans[2]],
        [0, 0, 0, 1]
    ])
    print("offset_mat:", np.array2string(offset_mat, separator=', '))
    trans_mat = trans.get_matrix()[0].cpu().numpy().T
    print("trans_mat:", np.array2string(trans_mat, separator=', '))
    scale_mat = scale_matrix.copy()
    print("scale_mat:", np.array2string(scale_mat, separator=', '))
    M = offset_mat @ trans_mat @ scale_mat
    print("final transformation:", np.array2string(M, separator=', '))

    # transform the original mesh using the final transformation
    vert_orig = np.array(mesh.vertices, dtype=np.float64)
    vert_orig = np.hstack((vert_orig, np.ones((vert_orig.shape[0], 1))))
    vert_trans = vert_orig @ M.T
    vert_trans = vert_trans[:, :3]
    final_mesh_data = {
        'vertices': vert_trans,
        'vertex_colors': (np.ones_like(vert_trans) * 255).astype(np.uint8)  # Ensure colors are in 0-255 range and uint8
    }
    final_mesh = trimesh.Trimesh(**final_mesh_data)
    final_mesh.export(os.path.join(out_dir, 'final.ply'))

    return M


if __name__ == "__main__":

    # Input file paths
    mask_img_path = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs/rope_double_hand/mask/0/1/0.png'
    raw_img_path = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs/rope_double_hand/color/0/0.png'
    depth_path = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs/rope_double_hand/depth/0/0.npy'
    pcd_file = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs/rope_double_hand/pcd/0/first_frame_object.npz'
    mesh_file = '/home/haoyuyh3/Documents/maxhsu/qqtt/mesh-recon/TRELLIS/output/rope_double_hand/sample.glb'
    metadata_file = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs/rope_double_hand/metadata.json'
    calibration_file = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs/rope_double_hand/calibrate.pkl'
    output_dir = './output'

    # Load metadata
    camera_id = 0
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    intrinsic = np.array(data["intrinsics"])[camera_id]
    c2ws = pickle.load(open(calibration_file, 'rb'))
    w2c = np.linalg.inv(c2ws[camera_id])

    # Get mask bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    mask_box = get_bbox(mask_img)
    print("Mask bounding box:", mask_box)

    # Estimate pose
    M = estimate_pose(
        mesh_file, pcd_file, raw_img_path, mask_img, mask_box, output_dir, intrinsic, w2c, depth_path,
        num_samples=8, num_ups=1
    )