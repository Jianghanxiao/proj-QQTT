import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg


def visualize_pc_real(
    object_points,
    object_colors,
    controller_points,
    object_visibilities=None,
    object_motions_valid=None,
    FPS=None,
    visualize=True,
    save_video=False,
    save_path=None,
):
    FPS = cfg.FPS if FPS is None else FPS

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors.shape[1] < object_points.shape[1]:
        # If the object_colors is not the same as object_points, fill the colors with black
        object_colors = np.concatenate(
            [
                object_colors,
                np.zeros(
                    (object_colors.shape[0], object_points.shape[1] - object_colors.shape[1], 3)
                ),
            ],
            axis=1,
        )

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

    controller_meshes = []
    prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_motions_valid is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_motions_valid[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_motions_valid[i])[0], :]
            )
        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01
                ).translate(origin)
                controller_mesh.paint_uniform_color(origin_color)
                controller_meshes.append(controller_mesh)
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

        # Capture frame and write to video file if save_video is True
        if save_video:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        if visualize:
            time.sleep(1 / FPS)

    vis.destroy_window()
    if save_video:
        video_writer.release()


def visualize_pc(
    pcs,
    FPS=None,
    visualize=True,
    save_video=False,
    save_path=None,
    springs=None,
    spring_params=None,
):
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

    if springs is not None:
        assert spring_params is not None
        assert len(spring_params) == len(springs)
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(pcs[0])
        lineset.lines = o3d.utility.Vector2iVector(springs)
        lineset.colors = o3d.utility.Vector3dVector(
            np.array([0.0, 1.0, 0.0]) * spring_params[:, None]
        )
        vis.add_geometry(lineset)

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
        if springs is not None:
            lineset.points = o3d.utility.Vector3dVector(pcs[i])
            vis.update_geometry(lineset)
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
