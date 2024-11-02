# Need to further process the collected data and process it
import os
import json

num_cameras = 3
output_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect"
calibrate_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/calibrate.pkl"

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data_collect"
case_name = "test_static"
# Need to manually control this for each video to cut (based on camera 0 always)
start_step = 150
end_step = 155


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    new_metadata = {}

    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)

    new_metadata["intrinsics"] = data["intrinsics"]
    new_metadata["serial_numbers"] = data["serial_numbers"]
    new_metadata["fps"] = data["fps"]
    new_metadata["WH"] = data["WH"]

    frame_num = 0
    step_timestamps = data["recording"]
    final_frames = []
    # Get all frames of camera 0 between start_step and end_step
    # And match the timestamps of other cameras
    for i in step_timestamps["0"].keys():
        if int(i) >= start_step and int(i) <= end_step:
            frame_num += 1
            timestamp = step_timestamps["0"][i]
            current_frame = [int(i)]
            for j in range(1, num_cameras):
                # Search the adjacent step_idx and get the most close timestamp
                potential_timestamps = []
                step_idx = int(i)
                min_diff = 10
                best_idx = None
                for k in range(-3, 4):
                    if str(step_idx + k) in step_timestamps[str(j)]:
                        diff = abs(
                            step_timestamps[str(j)][str(step_idx + k)] - timestamp
                        )
                        if diff < min_diff:
                            min_diff = diff
                            best_idx = step_idx + k
                current_frame.append(best_idx)
            final_frames.append(current_frame)

    new_metadata["frame_num"] = frame_num
    new_metadata["start_step"] = start_step
    new_metadata["end_step"] = end_step
    # Move the files into a final data format
    exist_dir(f"{output_path}")
    exist_dir(f"{output_path}/{case_name}")
    exist_dir(f"{output_path}/{case_name}/color")
    exist_dir(f"{output_path}/{case_name}/depth")
    for i in range(num_cameras):
        exist_dir(f"{output_path}/{case_name}/color/{i}")
        exist_dir(f"{output_path}/{case_name}/depth/{i}")

    os.system(f"cp {calibrate_path} {output_path}/{case_name}")

    with open(f"{output_path}/{case_name}/metadata.json", "w") as f:
        json.dump(new_metadata, f)
    for k, frame in enumerate(final_frames):
        for i in range(num_cameras):
            os.system(
                f"cp {base_path}/{case_name}/color/{i}/{frame[i]}.png {output_path}/{case_name}/color/{i}/{k}.png"
            )
            os.system(
                f"cp {base_path}/{case_name}/depth/{i}/{frame[i]}.npy {output_path}/{case_name}/depth/{i}/{k}.npy"
            )

    # for each camera, create a video for all frames with the fps 30
    # Useing opencv to create the video
    for i in range(num_cameras):
        os.system(
            f"ffmpeg -r 30 -start_number 0 -f image2 -i {output_path}/{case_name}/color/{i}/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {output_path}/{case_name}/color/{i}.mp4"
        )
