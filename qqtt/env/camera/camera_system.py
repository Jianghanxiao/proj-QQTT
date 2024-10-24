from realsense import MultiRealsense, SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


class CameraSystem:
    def __init__(self, WH=[640, 480], fps=30, num_cameras=4):
        self.WH = WH
        self.fps = fps

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        self.n_cameras = len(self.serial_numbers)
        assert (
            self.n_cameras == num_cameras
        ), f"Only {self.n_cameras} cameras are connected."

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense = MultiRealsense(
            serial_numbers=self.serial_numbers,
            shm_manager=self.shm_manager,
            resolution=(self.WH[0], self.WH[1]),
            capture_fps=self.fps,
            enable_color=True,
            enable_depth=True,
            process_depth=True,
            verbose=False,
        )
        # Some camera settings
        self.realsense.set_exposure(exposure=100, gain=60)
        self.realsense.set_white_balance(3800)

        self.realsense.start()
        time.sleep(3)

    def get_observation(self):
        # Used to get the latest observations from all cameras
        data = self._get_sync_frame()
        # TODO: Process the data when needed

    def _get_sync_frame(self, k=4):
        assert self.realsense.is_ready

        # Get the latest k frames from all cameras, and picked the latest synchronized frames
        last_realsense_data = self.realsense.get(k=k)
        timestamp_list = [x["timestamp"][-1] for x in last_realsense_data.values()]
        last_timestamp = np.min(timestamp_list)

        data = {}
        for camera_idx, value in last_realsense_data.items():
            this_timestamps = value["timestamp"]
            min_diff = 10
            best_idx = None
            for i, this_timestamp in enumerate(this_timestamps):
                diff = np.abs(this_timestamp - last_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            # remap key, step_idx is different, timestamp can be the same when some frames are lost
            data[camera_idx]["color"] = value["color"][best_idx]
            data[camera_idx]["depth"] = value["depth"][best_idx]
            data[camera_idx]["timestamp"] = value["timestamp"][best_idx]
            data[camera_idx]["step_idx"] = value["step_idx"][best_idx]

        return data


if __name__ == "__main__":
    camera_system = CameraSystem()
    camera_system._get_sync_frame()
