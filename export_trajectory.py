import os
import glob

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments"
output_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments_transfer"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

existDir(output_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    os.system(
        f"cp {base_path}/{case_name}/inference.pkl {output_path}/{case_name}/inference.pkl"
    )
    os.system(
        f"cp {base_path}/{case_name}/inference.mp4 {output_path}/{case_name}/inference.mp4"
    )

