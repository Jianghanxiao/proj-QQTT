from qqtt.data import SimpleData

class InvPhyTrainer:
    def __init__(self, data_path, device='cuda'):
        self.dataset = SimpleData(data_path, device=device)
        self.dataset.visualize()