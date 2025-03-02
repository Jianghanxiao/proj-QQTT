from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

# background_image = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data_collect/20250227_143503/color/0/204.png"
background_image = "/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/data_collect/20250227_143503/color/0/204.png"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian_data",
    )
    parser.add_argument("--case_name", type=str, default="single_lift_rope")
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument("--inv_ctrl", action="store_true", help="invert horizontal control direction")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"temp_experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    # with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
    #     c2ws = pickle.load(f)
    # w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    # cfg.c2ws = np.array(c2ws)
    # cfg.w2cs = np.array(w2cs)
    # with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
    #     data = json.load(f)
    # cfg.intrinsics = np.array(data["intrinsics"])
    # cfg.WH = data["WH"]
    cfg.overlay_path = f"/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/data_collect/20250227_143503/color"

    # TODO: my version of reading intrinsics and extrinsics
    with open(f"{base_path}/{case_name}/camera_meta.pkl", 'rb') as f:
        camera_info = pickle.load(f)
    c2ws = camera_info['c2ws']
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    cfg.intrinsics = np.array([np.array(intr) for intr in camera_info['intrinsics']])
    cfg.WH = (848, 480)  # fixed resolution
    exp_name = 'init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'
    gaussians_path = f"/home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/gaussian_splatting/output/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    cfg.mask_path = f"{base_path}/{case_name}/mask"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.interactive_playground(best_model_path, gaussians_path, args.n_ctrl_parts, args.inv_ctrl)


    # TODO: installation commands
    # pip install gsplat
    # export PATH=/home/haoyuyh3/Documents/maxhsu/cuda/cuda-12.1/bin:$PATH
    # export LD_LIBRARY_PATH=/home/haoyuyh3/Documents/maxhsu/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
    # cd gaussian_splatting/
    # pip install submodules/diff-gaussian-rasterization/
    # pip install submodules/simple-knn/
    # pip install kornia