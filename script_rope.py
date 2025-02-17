import os
import csv
import glob

base_path = "/data/proj-qqtt/processed_data/rope_variants"

os.system("rm -f timer.log")

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    category = "twine"
    shape_prior = "false"
    if shape_prior.lower() == "true":
        os.system(
            f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category} --shape_prior"
        )
    else:
        os.system(
            f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category}"
        )
