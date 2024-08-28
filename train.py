from qqtt import InvPhyTrainer, InvPhyTrainerCMA
from qqtt.utils import logger
from datetime import datetime


def demo_multiple_k():
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/TwoK_impulse_clamp_rand1"
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


def demo_cma_collision():
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/cma_collision_rand1"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerCMA(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/multiple_k_data_prepare/table_2k.npy",
        base_dir=base_dir,
    )
    trainer.optimize_collision(
        model_path="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments/TwoK_impulse_clamp_rand1/train/iter_499.pth"
    )


if __name__ == "__main__":
    # demo_multiple_k()
    # demo_rigid()
    demo_cma_collision()
