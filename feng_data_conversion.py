#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
import cv2
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_feng_calibration(calibration_path):
    """Load feng's calibration data and convert to c2w format"""
    # Load intrinsics
    intrinsics = np.load(os.path.join(calibration_path, "intrinsics.npy"))

    # Load extrinsics
    with open(os.path.join(calibration_path, "rvecs.pkl"), "rb") as f:
        rvecs_dict = pickle.load(f)
    with open(os.path.join(calibration_path, "tvecs.pkl"), "rb") as f:
        tvecs_dict = pickle.load(f)

    # Get serial numbers (camera IDs)
    serial_numbers = list(rvecs_dict.keys())

    # Convert to c2w matrices
    c2w_matrices = []
    intrinsics_list = []

    for i, serial in enumerate(serial_numbers):
        # Get rotation and translation vectors
        rvec = rvecs_dict[serial].flatten()
        tvec = tvecs_dict[serial].flatten()

        # Convert rotation vector to rotation matrix
        rot_mat = R.from_rotvec(rvec).as_matrix()

        # Create 4x4 transformation matrix (world to camera)
        w2c = np.eye(4)
        w2c[:3, :3] = rot_mat
        w2c[:3, 3] = tvec

        # Convert to camera to world
        c2w = np.linalg.inv(w2c)
        c2w_matrices.append(c2w)

        # Get intrinsics for this camera
        intrinsics_list.append(intrinsics[i].tolist())

    return c2w_matrices, intrinsics_list, serial_numbers


def convert_depth_to_npy(depth_png_path, output_npy_path, target_size=(480, 848)):
    """Convert depth PNG to NPY format with target size"""
    depth_img = cv2.imread(depth_png_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Warning: Could not load depth image {depth_png_path}")
        return

    # Resize to target size if needed
    if depth_img.shape[:2] != target_size:
        raise ValueError("depth size not correct")

    # Save as npy
    np.save(output_npy_path, depth_img)


def convert_rgb_to_png(rgb_jpg_path, output_png_path):
    """Convert RGB JPG to PNG format"""
    img = cv2.imread(rgb_jpg_path)
    if img is None:
        print(f"Warning: Could not load RGB image {rgb_jpg_path}")
        return
    cv2.imwrite(output_png_path, img)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="feng_data_collect/feng_rope_v6_0002",
        help="Path to feng's data directory",
    )
    parser.add_argument(
        "--output_path", type=str, default="data/feng_version", help="Output directory"
    )
    parser.add_argument(
        "--case_name",
        type=str,
        default="feng_rope_v6_0002",
        help="Name for the converted case",
    )
    parser.add_argument(
        "--start_frame", type=int, default=169, help="Start frame index"
    )
    parser.add_argument(
        "--end_frame", type=int, default=1000, help="End frame index (-1 for all frames)"
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    case_name = args.case_name
    start_frame = args.start_frame
    end_frame = args.end_frame

    # Load feng's calibration data
    calibration_path = os.path.join(input_path, "calibration")
    c2w_matrices, intrinsics_list, serial_numbers = load_feng_calibration(
        calibration_path
    )

    # Determine total frames
    camera_0_rgb_path = os.path.join(input_path, "camera_0", "rgb")
    total_frames = len([f for f in os.listdir(camera_0_rgb_path) if f.endswith(".jpg")])

    if end_frame == -1:
        end_frame = total_frames - 1

    print(f"Processing frames {start_frame} to {end_frame} (total: {total_frames})")

    # Create output directory structure
    output_case_path = os.path.join(output_path, case_name)
    exist_dir(output_case_path)
    exist_dir(os.path.join(output_case_path, "color"))
    exist_dir(os.path.join(output_case_path, "depth"))

    # Camera mapping: feng camera 1 -> our camera 0, feng camera 0 -> our camera 1, feng camera 2 -> our camera 2
    camera_mapping = {1: 0, 0: 1, 2: 2}

    for our_cam_id in range(3):
        exist_dir(os.path.join(output_case_path, "color", str(our_cam_id)))
        exist_dir(os.path.join(output_case_path, "depth", str(our_cam_id)))

    # Process frames
    frame_count = 0

    for frame_idx in range(start_frame, end_frame + 1):
        for feng_cam_id in range(3):
            our_cam_id = camera_mapping[feng_cam_id]

            # Process RGB
            rgb_input_path = os.path.join(
                input_path, f"camera_{feng_cam_id}", "rgb", f"{frame_idx:06d}.jpg"
            )
            rgb_output_path = os.path.join(
                output_case_path, "color", str(our_cam_id), f"{frame_count}.png"
            )

            if os.path.exists(rgb_input_path):
                convert_rgb_to_png(rgb_input_path, rgb_output_path)

            # Process Depth
            depth_input_path = os.path.join(
                input_path, f"camera_{feng_cam_id}", "depth", f"{frame_idx:06d}.png"
            )
            depth_output_path = os.path.join(
                output_case_path, "depth", str(our_cam_id), f"{frame_count}.npy"
            )

            if os.path.exists(depth_input_path):
                convert_depth_to_npy(depth_input_path, depth_output_path)

        frame_count += 1

    # Remap calibration data according to camera mapping
    remapped_c2w = [None] * 3
    remapped_intrinsics = [None] * 3
    remapped_serials = [None] * 3

    for feng_cam_id in range(3):
        our_cam_id = camera_mapping[feng_cam_id]
        remapped_c2w[our_cam_id] = c2w_matrices[feng_cam_id]
        remapped_intrinsics[our_cam_id] = intrinsics_list[feng_cam_id]
        remapped_serials[our_cam_id] = serial_numbers[feng_cam_id]

    # Save calibrate.pkl
    calibrate_output_path = os.path.join(output_case_path, "calibrate.pkl")
    with open(calibrate_output_path, "wb") as f:
        pickle.dump(remapped_c2w, f)

    # Create metadata.json
    metadata = {
        "intrinsics": remapped_intrinsics,
        "serial_numbers": remapped_serials,
        "fps": 30,
        "WH": [848, 480],
        "frame_num": end_frame - start_frame + 1,
        "start_step": start_frame,
        "end_step": end_frame
    }

    metadata_output_path = os.path.join(output_case_path, "metadata.json")
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f)

    # Create videos for each camera
    print("Creating videos...")
    for cam_id in range(3):
        color_dir = os.path.join(output_case_path, "color", str(cam_id))
        video_output = os.path.join(output_case_path, "color", f"{cam_id}.mp4")

        cmd = f"ffmpeg -y -r 30 -start_number 0 -f image2 -i {color_dir}/%d.png -vcodec libx264 -crf 0 -pix_fmt yuv420p {video_output}"
        os.system(cmd)

    print(f"Conversion completed! Output saved to: {output_case_path}")
    print(f"Processed {frame_count} frames")
    print(
        f"Camera mapping applied: feng_cam_1->our_cam_0, feng_cam_0->our_cam_1, feng_cam_2->our_cam_2"
    )


if __name__ == "__main__":
    main()
