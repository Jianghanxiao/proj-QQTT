import open3d as o3d
import numpy as np


def demo_visualize(data_path, n_frames):
    # Change from y up to z up
    transformation = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.io.read_point_cloud(f"{data_path}/000.ply")
    pcd.transform(transformation)
    vis.add_geometry(pcd)
    import pdb

    # Define ground plane vertices
    ground_vertices = np.array([[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0]])

    # Define ground plane triangular faces
    ground_triangles = np.array([[0, 2, 1], [0, 3, 2]])

    # Create Open3D mesh object
    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
    ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
    ground_mesh.paint_uniform_color([1, 211 / 255, 139 / 255])
    vis.add_geometry(ground_mesh)

    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0.5])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(3)

    for i in range(1, n_frames):
        file_name = f"{data_path}/{i:03d}.ply"
        new_pcd = o3d.io.read_point_cloud(file_name)
        new_pcd.transform(transformation)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(new_pcd.points))
        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    object = "duck"
    data_path = f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/spring_gaussian/mpm_synthetic/simulation/{object}"
    n_frames = 30

    demo_visualize(data_path, n_frames)
