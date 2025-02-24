import os
import json
import glob
import pickle
import numpy as np
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, default="./data/different_types")
parser.add_argument("--exp_path", type=str, default="./experiments")
parser.add_argument("--from_case", type=str, default="single_push_rope")
parser.add_argument("--to_case", type=str, default="rope_double_hand")
args = parser.parse_args()

base_path = args.base_path
exp_path = args.exp_path
from_case = args.from_case
to_case = args.to_case

OUTPUT_DIR = f"experiments_out_domain/{args.from_case}_to_{args.to_case}"

if __name__ == "__main__":
    with open(f"{OUTPUT_DIR}/final_points.pkl", "rb") as f:
        final_points = pickle.load(f)
    
    if "cloth" in from_case or "package" in from_case:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    print(f"[DATA TYPE]: {cfg.data_type}")

    # Read the first-satage optimized parameters
    optimal_path = f"experiments_optimization/{from_case}/optimal_params.pkl"
    assert os.path.exists(
        optimal_path
    ), f"{from_case}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{to_case}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{to_case}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{to_case}/color"

    logger.set_log_file(path=OUTPUT_DIR, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{from_case}/final_data.pkl",
        base_dir=OUTPUT_DIR,
        pure_inference_mode=True,
    )
    best_model_path = glob.glob(f"experiments/{from_case}/train/best_*.pth")[0]
    trainer.outdomain_inference(best_model_path, to_case_data_path=f"{base_path}/{to_case}/final_data.pkl", final_points=final_points)

