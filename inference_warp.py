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


# def test_real():
#     cfg.load_from_yaml("configs/real.yaml")
#     print(f"[DATA TYPE]: {cfg.data_type}")

#     # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     # base_dir = f"experiments/{current_time}"
#     base_dir = f"experiments/debug_rope_vis"
#     logger.set_log_file(path=base_dir, name="inv_phy_log")
#     trainer = InvPhyTrainerWarp(
#         data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect/rope_double_hand/final_data.pkl",
#         base_dir=base_dir,
#     )
#     trainer.test(
#         "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_exps/warp_rope_full/train/best_249.pth"
#     )


# def test_multiple_k():
#     cfg.load_from_yaml("configs/synthetic.yaml")
#     cfg.num_substeps = 1000
#     print(f"[DATA TYPE]: {cfg.data_type}")

#     # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     # base_dir = f"experiments/{current_time}"
#     base_dir = f"experiments/debug_table_vis"
#     logger.set_log_file(path=base_dir, name="inv_phy_log")
#     trainer = InvPhyTrainerWarp(
#         data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/multiple_k_data_prepare/table_2k.npy",
#         base_dir=base_dir,
#     )
#     trainer.test(
#         "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_exps/warp_table_full/train/best_499.pth"
#     )


if __name__ == "__main__":
    # test_real()
    # test_multiple_k()

    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    assert len(glob.glob(f"{base_dir}/train/best_*.pth")) > 0
    best_model_path = glob.glob(f"{base_dir}/train/best_*.pth")[0]
    trainer.test(best_model_path)
