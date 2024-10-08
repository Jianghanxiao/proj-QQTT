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
    base_dir = f"experiments/table_check_full_debug"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/multiple_k_data_prepare/table_2k.npy",
        base_dir=base_dir,
    )
    trainer.test(
        "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_exps/full_collision_fix/table_check_full/train/best_499.pth",
        normalization_factor=1e5,
    )


if __name__ == "__main__":
    demo_multiple_k()
