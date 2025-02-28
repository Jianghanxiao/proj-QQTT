import os
import glob

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
output_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_gt_track"

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

exist_dir(output_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}")

    exist_dir(f"{output_path}/{case_name}")
    # Copy the video to the output_path
    os.system(f"cp {dir_name}/color/0.mp4 {output_path}/{case_name}/0.mp4")

    

