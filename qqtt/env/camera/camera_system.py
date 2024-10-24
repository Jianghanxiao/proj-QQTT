from .realsense.multi_realsense import MultiRealsense, SingleRealsense
from multiprocessing.managers import SharedMemoryManager


class CameraSystem:
    def __init__(self, WH=[640, 480], fps=60):
        self.WH = WH
        self.fps = fps
