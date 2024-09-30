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
        self.init_spring_Y = 3e3
        self.init_collide_elas = 0.5
        self.init_collide_fric = 0.3
        self.collide_object_elas = 0.7
        self.collide_object_fric = 0.3

        self.radius = 0.1
        self.max_neighbours = 20

        self.num_substeps = 100

        # Parameters on whether update the collision parameters
        self.collision_learn = True

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }


cfg = SimpleConfig()
