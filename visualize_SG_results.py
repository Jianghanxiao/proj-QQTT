import glob
import json
import cv2
import os

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
prediction_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/baselines/Spring-Gaus/exp"
human_mask_path = (
    "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_human_mask"
)
object_mask_path = (
    "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/render_eval_data"
)

height, width = 480, 848
FPS = 30
alpha = 0.7

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    if not os.path.exists(f"{prediction_dir}/{case_name}/evaluations/0"):
        continue
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]

     # Need to prepare the video
    for i in range(3):
        # Process each camera
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(
            f"{prediction_dir}/{case_name}/{i}_integrate.mp4",
            fourcc,
            FPS,
            (width, height),
        )

        for frame_idx in range(frame_len):
            render_path = f"{prediction_dir}/{case_name}/evaluations/{i}/images_pred/{i}_{frame_idx:02d}.png"
            origin_image_path = f"{base_path}/{case_name}/color/{i}/{frame_idx}.png"
            human_mask_image_path = (
                f"{human_mask_path}/{case_name}/mask/{i}/0/{frame_idx}.png"
            )
            object_image_path = (
                f"{object_mask_path}/{case_name}/mask/{i}/{frame_idx}.png"
            )

            render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
            origin_img = cv2.imread(origin_image_path)
            human_mask = cv2.imread(human_mask_image_path)
            human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
            human_mask = human_mask > 0
            object_mask = cv2.imread(object_image_path)
            object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
            object_mask = object_mask > 0

            render_mask = (render_img != 0).any(axis=2)
            render_img[~render_mask] = 255

            final_image = origin_img.copy()
            final_image[render_mask] = render_img[render_mask]
            final_image[object_mask] = (1 - alpha) * render_img[
                object_mask
            ] + alpha * origin_img[object_mask]
            final_image[human_mask] = origin_img[human_mask]

            video_writer.write(final_image)

        video_writer.release()