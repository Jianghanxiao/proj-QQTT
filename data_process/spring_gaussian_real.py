import numpy as np
import json
import cv2
import os

# Resize Dimension: based on the requirement of DepthCrafter
resized_height = 576
resized_width = 1024


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_camera_info(
    data_path, save_path, height, width, resized_height, resized_width
):
    # # Camera intrinsic from the static periord, not useful for my case
    # camdata = read_cameras_binary(f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/spring_gaussian/real_capture/static/colmap/potato/cameras.bin")
    # H, W = 1080, 1920
    # H_S, W_S = 2672, 4752
    # K = np.array([
    #         [camdata[1].params[0] * W / W_S, 0, W / 2],
    #         [0, camdata[1].params[1] * H / H_S, H / 2],
    #         [0, 0, 1],
    #     ])

    camera_calib_path = f"{data_path}/dynamic/cameras_calib.json"
    with open(camera_calib_path, "r") as f:
        camera_calib = json.load(f)

    sx = resized_width / width
    sy = resized_height / height

    # Modify the intrinsic based on the resize of the image
    intrinsic = camera_calib["camera_matrix"]
    intrinsic = np.array(intrinsic)
    intrinsic[0, 0] *= sx
    intrinsic[1, 1] *= sy
    intrinsic[0, 2] = resized_width / 2
    intrinsic[1, 2] = resized_height / 2

    distortion = camera_calib["distortion_coefficient"]

    camera_calib.pop("camera_matrix")
    camera_calib.pop("distortion_coefficient")

    cameras = list(camera_calib.keys())
    c2ws = {}
    for camera in cameras:
        rvecs = camera_calib[camera]["rvecs"]
        tvecs = camera_calib[camera]["tvecs"]
        rot_mat, _ = cv2.Rodrigues(np.array(rvecs))
        w2c = np.eye(4)
        w2c[:3, :3] = rot_mat
        w2c[:3, 3] = np.array(tvecs).squeeze()
        c2w = np.linalg.inv(w2c)
        c2ws[camera] = c2w.tolist()

    with open(f"{save_path}/camera_info.json", "w") as f:
        json.dump(
            {"intrinsic": intrinsic.tolist(), "distortion": distortion, "c2ws": c2ws}, f
        )

    return intrinsic, distortion, c2ws


def save_resize_images(object, data_path, save_path, config, cameras, FPS):
    global resized_width, resized_height

    exist_dir(f"{save_path}/imgs")
    exist_dir(f"{save_path}/videos")
    # Also save the images into videos
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
    video_writer_whole = cv2.VideoWriter(
        f"{save_path}/videos/whole.mp4", fourcc, FPS, (resized_width, resized_height)
    )

    for camera in cameras:
        exist_dir(f"{save_path}/imgs/{camera}")
        video_writer_each = cv2.VideoWriter(
            f"{save_path}/videos/{camera}.mp4",
            fourcc,
            FPS,
            (resized_width, resized_height),
        )
        img_names = config[camera]
        index = 0
        for img_name in img_names:
            img = cv2.imread(f"{data_path}/dynamic/videos_images/{camera}/{img_name}")
            img = cv2.resize(img, (resized_width, resized_height))
            cv2.imwrite(f"{save_path}/imgs/{camera}/{index}.jpg", img)
            # Write the images into videos
            video_writer_whole.write(img)
            video_writer_each.write(img)
            index += 1

        video_writer_each.release()
    video_writer_whole.release()


# Read the images
def reform_data(object, data_path, save_path, height=1080, width=1920, FPS=120):
    global resized_height, resized_width
    # Read the configs
    with open(f"{data_path}/dynamic/sequences/{object}/0.json", "r") as f:
        config = json.load(f)
    hit_frame = config["hit_frame"]
    config.pop("hit_frame")
    cameras = list(config.keys())

    save_resize_images(object, data_path, save_path, config, cameras, FPS)

    save_camera_info(
        data_path,
        save_path,
        height,
        width,
        resized_height=resized_height,
        resized_width=resized_width,
    )


if __name__ == "__main__":
    object = "burger"
    data_path = f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/spring_gaussian/real_capture"

    save_path = (
        f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reform_SG/{object}"
    )
    exist_dir(save_path)
    # Collect the annotations into my customized format
    reform_data(object, data_path, save_path)
