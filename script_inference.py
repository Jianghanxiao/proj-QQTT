import glob
import os
import json

base_path = "./data/more_clothes"
dir_names = glob.glob(f"experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )
