from qqtt import InvPhyTrainer, InvPhyTrainerCam
from qqtt.utils import logger
from datetime import datetime


# Demo to inverse physics for the simple 3D tracking data
def demo_simple(springY):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/SingleK_{springY}"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainer(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/simple_data_prepare/table_{springY}.0.npy",
        base_dir=base_dir,
    )
    trainer.train()

# Demo to inverse physics for the simple 3D tracking data
def demo_cma(springY):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{current_time}"
    base_dir = f"experiments/SingleK_{springY}"
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerCam(
        data_path=f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data_prepare/simple_data_prepare/table_{springY}.0.npy",
        base_dir=base_dir,
    )
    trainer.train()


if __name__ == "__main__":
    # demo_simple(springY=3000)
    # demo_simple(springY=30000)
    # demo_simple(springY=300000)
    # demo_simple(springY=3000000)
    demo_cma(springY=3000)
