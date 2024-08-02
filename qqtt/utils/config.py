from .misc import singleton


@singleton
class SimpleConfig:
    def __init__(self):
        self.FPS = 10
        self.dt = 5e-5
        self.num_substeps = 1000
        self.dashpot_damping = 100
        self.drag_damping = 3
        self.base_lr = 1e-3
        self.iterations = 500
        self.vis_interval = self.iterations / 10
        self.init_spring_Y = 3e4
        self.init_collide_elas = 1
        self.init_collide_fric = 0.3

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }


cfg = SimpleConfig()
