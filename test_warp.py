from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch


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


def test_real():
    cfg.load_from_yaml("configs/real.yaml")
    cfg.visualize_ground = True
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/debug_rope_vis"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect/rope_double_hand/final_data.pkl",
        base_dir=base_dir,
    )
    trainer.test(
        "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_exps/warp_rope_full/train/best_249.pth"
    )


def test_multiple_k():
    cfg.load_from_yaml("configs/synthetic.yaml")
    cfg.num_substeps = 1000
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/debug_table_vis"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/multiple_k_data_prepare/table_2k.npy",
        base_dir=base_dir,
    )
    trainer.test(
        "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_exps/warp_table_full/train/best_499.pth"
    )

if __name__ == "__main__":
    test_real()
    # test_multiple_k()