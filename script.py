import os
import glob

case_names = []
# base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
base_path = "/data/proj-qqtt/processed_data/rope_variants"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    # if case_name == "rope_double_hand" or case_name == "double_lift_frog":
    #     continue
    case_names.append(dir_name.split("/")[-1])

# for case_name in case_names:
#     print(f"Processing {case_name}")
#     os.system(f"python data_process/data_process_pcd.py --case_name {case_name}")

# for case_name in case_names:
#     os.system(f"python data_process/data_process_mask.py --case_name {case_name}")

# for case_name in case_names:
#     os.system(f"python data_process/get_track.py --case_name {case_name}")

# for case_name in case_names:
#     os.system(f"python data_process/data_process_track.py --case_name {case_name} &")

# for case_name in case_names:
#     os.system(f"python data_process/data_process_sample.py --case_name {case_name}")

if True:
    # Quick code to verify all the data
    import pickle
    import open3d as o3d
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import json


    def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
        sphere.paint_uniform_color(color)
        return sphere


    def visualize_track(track_data, save_video=False, save_path=None, FPS=30):
        object_points = track_data["object_points"]
        object_colors = track_data["object_colors"]
        object_visibilities = track_data["object_visibilities"]
        object_motions_valid = track_data["object_motions_valid"]
        controller_points = track_data["controller_points"]

        frame_num = object_points.shape[0]

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=not save_video)
        controller_meshes = []
        prev_center = []

        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

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

        for i in range(frame_num):
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                rainbow_colors[np.where(object_visibilities[i])[0]]
            )

            if i == 0:
                render_object_pcd = object_pcd
                vis.add_geometry(render_object_pcd)
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_meshes.append(
                        getSphereMesh(origin, color=origin_color, radius=0.01)
                    )
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

        vis.destroy_window()
        if save_video:
            video_writer.release()


    # base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
    base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/rope_variants"

    for case_name in case_names:
        print(f"Visualizing {case_name}")
        with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
            track_data = pickle.load(f)
        # with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        #     track_data = pickle.load(f)

        # visualize_track(
        #     track_data, save_video=True, save_path=f"{base_path}/{case_name}/test_track_data.mp4"
        # )
        visualize_track(
            track_data, save_video=True, save_path=f"{base_path}/{case_name}/whole.mp4"
        )

        frame_len = track_data["object_points"].shape[0]

        dir = {}
        dir["frame_len"] = frame_len
        dir["train"] = [0, int(frame_len * 0.7)]
        dir["test"] = [int(frame_len * 0.7) + 1, frame_len]
        with open(f"{base_path}/{case_name}/split.json", "w") as f:
            json.dump(dir, f)

# import glob

# def exist_dir(dir):
#     if not os.path.exists(dir):
#         os.makedirs(dir)

# base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/rope_variants"
# exist_dir(base_path)

# dir_names = glob.glob(f"/data/proj-qqtt/processed_data/rope_variants/*")
# for dir_name in dir_names:
#     case_name = dir_name.split("/")[-1]
#     exist_dir(f"{base_path}/{case_name}")
#     os.system(f"cp -r {dir_name}/final_data.pkl {base_path}/{case_name}")
