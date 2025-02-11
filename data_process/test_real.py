# Process to get the masks of the controller and the object

import os
from argparse import ArgumentParser

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
parser = ArgumentParser()
parser.add_argument("--case_name", type=str, default="single_put_rope")
args = parser.parse_args()
case_name = args.case_name
print(f"Processing {case_name}")
TEXT_PROMPT = "twine.hand"
camera_num = 1

for camera_idx in range(camera_num):
    print(f"python real_usage.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}")
    os.system(f"python ./data_process/real_usage.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}")
    os.system(f"rm -rf real_data/{case_name}/{camera_idx}")

# for i in range(1, 30):
#     case_name = f"rope_{i}"
#     for camera_idx in range(camera_num):
#         print(f"python real_usage.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}")
#         os.system(f"python real_usage.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}")
#         os.system(f"rm -rf real_data/{case_name}/{camera_idx}")