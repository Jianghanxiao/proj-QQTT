from qqtt import RealInvPhyTrainer
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
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/rope_double_hand"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = RealInvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect/rope_double_hand/final_data.pkl",
        base_dir=base_dir,
    )
    trainer.train()
    # trainer.test("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/rigid/train/iter_199.pth")


if __name__ == "__main__":
    demo_real()
