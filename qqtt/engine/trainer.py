from qqtt.data import SimpleData
from qqtt.utils import logger

class InvPhyTrainer:
    def __init__(self, data_path, base_dir, device='cuda:0'):
        self.dataset = SimpleData(data_path, base_dir, device=device, visualize=True)