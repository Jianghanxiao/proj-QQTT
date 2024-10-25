from .realsense import MultiRealsense, SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
from pynput import keyboard
import cv2
import json
import os

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class CameraSystem:
    def __init__(self, WH=[640, 480], fps=30, num_cam=4):
        self.WH = WH
        self.fps = fps

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        self.num_cam = len(self.serial_numbers)
        assert self.num_cam == num_cam, f"Only {self.num_cam} cameras are connected."

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
        self.recording = False
        self.end = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("Camera system is ready.")

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

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if self.recording == False:
                    self.recording = True
                    print("Start recording")
                else:
                    self.recording = False
                    self.end = True
        except AttributeError:
            pass

    def record(self, output_path):
        exist_dir(output_path)
        exist_dir(f"{output_path}/color")
        exist_dir(f"{output_path}/depth")

        metadata = {}
        intrinsics = self.realsense.get_intrinsics()
        metadata["intrinsics"] = intrinsics.tolist()
        metadata["serial_numbers"] = self.serial_numbers
        metadata["fps"] = self.fps
        metadata["WH"] = self.WH
        metadata["recording"] = {}
        for i in range(self.num_cam):
            metadata["recording"][i] = {}
            exist_dir(f"{output_path}/color/{i}")
            exist_dir(f"{output_path}/depth/{i}")

        # Set the max time for recording
        last_step_idxs = [-1] * self.num_cam
        while not self.end:
            if self.recording:
                last_realsense_data = self.realsense.get()
                timestamps = [
                    last_realsense_data[i]["timestamp"].item()
                    for i in range(self.num_cam)
                ]
                step_idxs = [
                    last_realsense_data[i]["step_idx"].item()
                    for i in range(self.num_cam)
                ]

                if not all(
                    [step_idxs[i] == last_step_idxs[i] for i in range(self.num_cam)]
                ):
                    for i in range(self.num_cam):
                        if last_step_idxs[i] != step_idxs[i]:
                            # Record the the step for this camera
                            time_stamp = timestamps[i]
                            step_idx = step_idxs[i]
                            color = last_realsense_data[i]["color"]
                            depth = last_realsense_data[i]["depth"]

                            metadata["recording"][i][step_idx] = time_stamp
                            cv2.imwrite(
                                f"{output_path}/color/{i}/{step_idx}.png", color
                            )
                            np.save(f"{output_path}/depth/{i}/{step_idx}.npy", depth)

        print("End recording")
        self.listener.stop()
        with open(f"{output_path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        self.realsense.stop()
