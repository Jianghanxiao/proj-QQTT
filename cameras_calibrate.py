from qqtt.env import CameraSystem

if __name__ == "__main__":
    # camera_system = CameraSystem()
    camera_system = CameraSystem(WH=[1280, 720], fps=5)
    camera_system.calibrate()