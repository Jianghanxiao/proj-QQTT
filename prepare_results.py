import os
import glob

base_path = "experiments"
output_dir = "experiments_transfer"
os.makedirs(output_dir, exist_ok=True)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    os.makedirs(f"{output_dir}/{case_name}", exist_ok=True)

    # Copy the log file
    os.system(f"cp {base_path}/{case_name}/inv_phy_log.log {output_dir}/{case_name}/inv_phy_log.log")
    # Copy the best model and the best video
    os.makedirs(f"{output_dir}/{case_name}/train", exist_ok=True)
    os.system(f"cp {base_path}/{case_name}/train/init.mp4 {output_dir}/{case_name}/train/init.mp4")
    best_model_name = glob.glob(f"{base_path}/{case_name}/train/best_*.pth")[0]
    best_iter = best_model_name.split("/")[-1].split("_")[-1].split(".")[0]
    os.system(f"cp {base_path}/{case_name}/train/best_{best_iter}.pth {output_dir}/{case_name}/train/best_{best_iter}.pth")
    os.system(f"cp {base_path}/{case_name}/train/sim_iter{best_iter}.mp4 {output_dir}/{case_name}/train/sim_iter{best_iter}.mp4")
