from .misc import singleton


@singleton
class SimpleConfig:
    def __init__(self):
        self.FPS = 10
        self.dt = 5e-5
        self.num_substeps = 100
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

        # Parameters on whether update the collision parameters
        self.collision_learn = True

        # Parameters on whether the gt point cloud is ordered across time
        self.match = True

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }


@singleton
class RealConfig:
    def __init__(self):
        self.FPS = 30
        self.dt = 5e-5
        self.num_substeps = round(1.0 / self.FPS / self.dt)
        self.dashpot_damping = 100
        self.drag_damping = 3
        self.base_lr = 1e-3
        self.iterations = 500
        self.vis_interval = 10
        self.init_spring_Y = 3e3
        self.init_collide_elas = 0.5
        self.init_collide_fric = 0.3
        self.collide_object_elas = 0.7
        self.collide_object_fric = 0.3

        self.chamfer_weight = 1
        self.track_weight = 1

        self.radius = 0.01
        self.max_neighbours = 20

        self.spring_Y_min = 1e3
        self.spring_Y_max = 1e5

        self.second_stage_iter = 50

        # Parameters on whether update the collision parameters
        self.collision_learn = True

        # DEBUG mode: set use_graph to False
        self.use_graph = True

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }


# cfg = SimpleConfig()
cfg = RealConfig()
