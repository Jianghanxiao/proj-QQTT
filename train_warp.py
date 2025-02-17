from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser


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
    base_dir = f"experiments/debug_warp_rope_full"
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


def demo_cloth():
    cfg.load_from_yaml("configs/cloth.yaml")
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/debug_test_cloth_no_shape_completion_3e4_0.02_30_more_radius_self_collision_0.02"
    # cfg.init_spring_Y = 3e3
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types/single_lift_cloth/final_data.pkl",
        base_dir=base_dir,
    )
    trainer.train()

def demo_package():
    cfg.load_from_yaml("configs/cloth.yaml")
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/debug_package"
    # cfg.init_spring_Y = 3e4
    # cfg.radius = 0.02
    # cfg.max_neighbours = 30
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types/weird_package/final_data.pkl",
        base_dir=base_dir,
    )
    trainer.train()

def demo_multiple_k():
    cfg.load_from_yaml("configs/synthetic.yaml")
    print(f"[DATA TYPE]: {cfg.data_type}")

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/warp_table_full"
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

    base_dir = f"experiments/billiard_warp_quick_0.06"
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
    # demo_real()
    # demo_multiple_k()
    # demo_billiard()
    # demo_cloth()
    # demo_package()

    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True
    )
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"
    cfg.init_spring_Y = 3e4

    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
    )
    trainer.train()
