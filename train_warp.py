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


def demo_real():
    cfg.load_from_yaml("configs/real.yaml")
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/debug_warp_rope_acc"
    cfg.spring_Y_min = 0
    # cfg.init_spring_Y = 1e3
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect/rope_double_hand/final_data.pkl",
        base_dir=base_dir,
    )
    trainer.train()
    # trainer.resume_train(
    #     model_path="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/rope_double_hand_clamp_more_control_smooth_a/train/iter_40.pth"
    # )
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/debug_warp/train/iter_0.pth")


def demo_multiple_k():
    cfg.load_from_yaml("configs/synthetic.yaml")
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/debug_warp_ground_collision"
    cfg.num_substeps = 1000
    cfg.init_spring_Y = 3e4
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/multiple_k_data_prepare/table_2k.npy",
        base_dir=base_dir,
    )
    # trainer.visualize_sim(save_only=False)
    trainer.train()
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/TwoK/train/iter_199.pth")


def demo_billiard():
    cfg.load_from_yaml("configs/synthetic.yaml")
    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/debug_billiard_warp_quick_0.06"
    cfg.iterations = 1000
    cfg.vis_interval = 50
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard.npy",
        mask_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_mask.npy",
        velocity_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_velocities.npy",
        base_dir=base_dir,
    )
    trainer.train()
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/billiard_initial_3e3_chamfer/train/iter_199.pth")


if __name__ == "__main__":
    demo_real()
    # demo_multiple_k()
    # demo_billiard()
