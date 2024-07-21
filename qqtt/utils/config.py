from .misc import singleton

@singleton
class SimpleConfig():
    def __init__(self):
        self.FPS = 10
        self.dt = 5e-5
        self.num_substeps = 1000
        self.dashpot_damping = 100
        self.drag_damping = 3
    

cfg = SimpleConfig()