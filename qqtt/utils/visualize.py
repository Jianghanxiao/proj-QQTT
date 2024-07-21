import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg


def visualize_pc(pcs, FPS=None, visualize=True, save_video=False, save_path=None):
    """
    Visualizes a sequence of point clouds and optionally saves it as a video.

    Args:
        pcs (numpy.ndarray or torch.Tensor): A 4D point cloud array with shape (n_frames, n_points, 3).
        FPS (int, optional): Frames per second for visualization. Default is None.
        visualize (bool, optional): Flag to display the visualization window. Default is True.
        save_video (bool, optional): Flag to save the visualization as a video. Default is False.
        save_path (str, optional): Path to save the video if save_video is True. Default is None.

    """
    FPS = cfg.FPS if FPS is None else FPS

    # Convert the pcs to numpy if it's tensor
    if isinstance(pcs, torch.Tensor):
        pcs = pcs.cpu().numpy()
    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visualize)

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    # Initialize video writer if save_video is True
    if save_video:
        # Create a dummy frame to get the width and height
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape

        if height <= 0 or width <= 0:
            raise ValueError(
                "Invalid dimensions for the video. Check the frame capture."
            )

        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcs[0])
    pcd.paint_uniform_color([1, 0, 0])
    vis.add_geometry(pcd)

    # Define ground plane vertices
    ground_vertices = np.array([[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0]])
    # Define ground plane triangular faces
    ground_triangles = np.array([[0, 2, 1], [0, 3, 2]])
    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
    ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
    ground_mesh.paint_uniform_color([1, 211 / 255, 139 / 255])
    vis.add_geometry(ground_mesh)

    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0.5])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(3)

    for i in range(1, pcs.shape[0]):
        pcd.points = o3d.utility.Vector3dVector(pcs[i])
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Capture frame and write to video file if save_video is True
        if save_video:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)  # Convert to 8-bit image
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        if visualize:
            time.sleep(1 / FPS)

    vis.destroy_window()
    if save_video:
        video_writer.release()
