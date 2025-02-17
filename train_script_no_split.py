import glob
import os

base_path = "./data/different_types_final"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    os.system(
        f"python train_warp.py --base_path {base_path} --case_name {case_name}"
    )