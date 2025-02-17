import glob
import os

# base_path = "/data/proj-qqtt/processed_data/rope_variants"
# output_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/rope_variants_final"
base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
output_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_final"

ADD_VIS = False

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

existDir(output_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    existDir(f"{output_path}/{case_name}")
    # Copy the final data
    os.system(f"cp {dir_name}/final_data.pkl {output_path}/{case_name}/final_data.pkl")
    # Copy the split data
    os.system(f"cp {dir_name}/split.json {output_path}/{case_name}/split.json")
    if ADD_VIS:
        os.system(f"cp {dir_name}/final_data.mp4 {output_path}/{case_name}/final_data.mp4")
