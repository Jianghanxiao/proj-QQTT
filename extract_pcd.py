import trimesh
import open3d as o3d
import numpy as np
import copy

table_path = "/home/hanxiao/Desktop/Research/proj-roboexp/RoboExp_data/assets/objects/models/table/92db5e13fa0c4c27a2689b962fc6305f.glb"
teddy_path = "/home/hanxiao/Desktop/Research/proj-roboexp/RoboExp_data/assets/objects/models/teddy/9ddb37ab928246a2acfb4e50c558c49c.glb"



if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh(teddy_path)
    pcd = mesh.sample_points_uniformly(number_of_points=20000)
    teddy = pcd.farthest_point_down_sample(1024)
    teddy.scale(0.01, center=[0,0,0])

    mesh = o3d.io.read_triangle_mesh(table_path)
    pcd = mesh.sample_points_uniformly(number_of_points=20000)
    table = pcd.farthest_point_down_sample(1024)
    table.scale(0.1, center=[0,0,0])

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([teddy, table, coordinate])

    # Save the teddy and table point clouds
    o3d.io.write_point_cloud("data/teddy.ply", teddy)
    o3d.io.write_point_cloud("data/table.ply", table)
