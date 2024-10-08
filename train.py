from qqtt import InvPhyTrainer, InvPhyTrainerCMA
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


def demo_multiple_k():
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    cfg.num_substeps = 1000
    cfg.init_spring_Y = 3e4
    base_dir = f"experiments/table_test_two_k"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/multiple_k_data_prepare/table_2k.npy",
        base_dir=base_dir,
    )
    # trainer.visualize_sim(save_only=False)
    trainer.train()
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/TwoK/train/iter_199.pth")


def demo_rigid():
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/rigid"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/rigid_data_prepare/table_rigid.npy",
        base_dir=base_dir,
    )
    trainer.train()
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/rigid/train/iter_199.pth")


def demo_billiard():
    base_dir = f"experiments/billiard_initial_3e3_fix_collision_toi_no_gradient"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard.npy",
        mask_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_mask.npy",
        velocity_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_velocities.npy",
        base_dir=base_dir,
    )
    trainer.train()
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/billiard/train/iter_199.pth")

def demo_billiard_continue():
    base_dir = f"experiments/billiard_initial_3e3_fix_collision_toi_continue"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    cfg.iterations = 1000
    trainer = InvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard.npy",
        mask_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_mask.npy",
        velocity_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_velocities.npy",
        base_dir=base_dir,
    )
    trainer.resume_train(model_path="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/billiard_initial_3e3_fix_collision_toi/train/iter_499.pth")
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/billiard/train/iter_199.pth")


def demo_cma_collision():
    base_dir = f"experiments/cma_collision_non_learned_collision"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerCMA(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard.npy",
        mask_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_mask.npy",
        velocity_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_velocities.npy",
        base_dir=base_dir,
    )
    cfg.model_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_exps/full_collision/billiard_initial_3e3_not_learn_collision/train/best_350.pth"
    trainer.optimize_collision(model_path=cfg.model_path)


if __name__ == "__main__":
    demo_multiple_k()
    # demo_rigid()
    # demo_billiard()
    # demo_cma_collision()
    # demo_billiard_continue()
