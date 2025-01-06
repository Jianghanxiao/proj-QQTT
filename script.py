import os

# for i in range(6, 30):
#     os.system(f"python data_process/data_process_mask.py --case_name rope_{i}")

# for i in range(2, 30):
#     os.system(f"python data_process/get_track.py --case_name rope_{i}")

# for i in range(1, 30):
#     os.system(f"python data_process/data_process_track.py --case_name rope_{i} &")

# for i in range(1, 30):
#     print(f"Processing rope_{i}")
#     os.system(f"python data_process/data_process_sample.py --case_name rope_{i}")


# # Quick code to verify all the data
# import pickle
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt


# def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
#     sphere.paint_uniform_color(color)
#     return sphere


# def visualize_track(track_data):
#     object_points = track_data["object_points"]
#     object_colors = track_data["object_colors"]
#     object_visibilities = track_data["object_visibilities"]
#     object_motions_valid = track_data["object_motions_valid"]
#     controller_points = track_data["controller_points"]

#     frame_num = object_points.shape[0]

#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     controller_meshes = []
#     prev_center = []

#     y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
#     y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
#     rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

#     for i in range(frame_num):
#         object_pcd = o3d.geometry.PointCloud()
#         object_pcd.points = o3d.utility.Vector3dVector(
#             object_points[i, np.where(object_motions_valid[i])[0], :]
#         )
#         # object_pcd.colors = o3d.utility.Vector3dVector(
#         #     object_colors[i, np.where(object_motions_valid[i])[0], :]
#         # )
#         object_pcd.colors = o3d.utility.Vector3dVector(
#             rainbow_colors[np.where(object_motions_valid[i])[0]]
#         )

#         if i == 0:
#             render_object_pcd = object_pcd
#             vis.add_geometry(render_object_pcd)
#             # Use sphere mesh for each controller point
#             for j in range(controller_points.shape[1]):
#                 origin = controller_points[i, j]
#                 origin_color = [1, 0, 0]
#                 controller_meshes.append(
#                     getSphereMesh(origin, color=origin_color, radius=0.01)
#                 )
#                 vis.add_geometry(controller_meshes[-1])
#                 prev_center.append(origin)
#             # Adjust the viewpoint
#             view_control = vis.get_view_control()
#             view_control.set_front([1, 0, -2])
#             view_control.set_up([0, 0, -1])
#             view_control.set_zoom(1)
#         else:
#             render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
#             render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
#             vis.update_geometry(render_object_pcd)
#             for j in range(controller_points.shape[1]):
#                 origin = controller_points[i, j]
#                 controller_meshes[j].translate(origin - prev_center[j])
#                 vis.update_geometry(controller_meshes[j])
#                 prev_center[j] = origin
#             vis.poll_events()
#             vis.update_renderer()


# base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/rope_variants"

# bad_data = [15, 20, 27, 28, 29]

# for i in range(1, 30):
#     if i in bad_data:
#         continue
#     print(f"Visualizing rope_{i}")
#     case_name = f"rope_{i}"
#     with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
#         track_data = pickle.load(f)

#     visualize_track(track_data)
