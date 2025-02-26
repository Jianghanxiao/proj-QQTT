import glob
import os

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"
output_path = (
    "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_SG"
)

ADD_VIS = False


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


existDir(output_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    # Copy the final data
    os.system(f"cp {dir_name}/final_data.pkl {output_path}/{case_name}/final_data.pkl")
    # Copy the split data
    os.system(f"cp {dir_name}/split.json {output_path}/{case_name}/split.json")

    # Copy the color dir and the mask dir
    os.system(f"cp -r {dir_name}/color {output_path}/{case_name}/")
    os.system(f"cp -r {dir_name}/mask {output_path}/{case_name}/")
    existDir(f"{output_path}/{case_name}/pcd")
    os.system(f"cp {dir_name}/pcd/0.npz {output_path}/{case_name}/pcd/0.npz")

    # Copy the intrinsic and extrinsic parameters
    os.system(f"cp {dir_name}/calibrate.pkl {output_path}/{case_name}/calibrate.pkl")
    os.system(f"cp {dir_name}/metadata.json {output_path}/{case_name}/metadata.json")
