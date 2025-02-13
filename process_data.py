import os
from argparse import ArgumentParser
import time
import logging
import json

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
args = parser.parse_args()

# Set the debug flags
PROCESS_SEG = False
PROCESS_SHAPE_PRIOR = False
PROCESS_TRACK = False
PROCESS_OTHER = False

base_path = args.base_path
case_name = args.case_name
category = args.category
TEXT_PROMPT = f"{category}.hand"
CONTROLLER_NAME = "hand"

logger = None


def setup_logger(log_file="timer.log"):
    global logger  # 声明全局变量

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        # 避免重复添加 handler
        if not logger.handlers:
            # 创建文件 handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            # 创建终端 handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SEG:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        os.system(
            f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
        )


if PROCESS_SHAPE_PRIOR:
    # Get the mask path for the image
    with open(f"{base_path}/{case_name}/mask/mask_info_{0}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_path = f"{base_path}/{case_name}/mask/0/{obj_idx}/0.png"

    existDir(f"{base_path}/{case_name}/shape")
    # Get the high-resolution of the image to prepare for the trellis generation
    with Timer("Image Upscale"):
        os.system(
            f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --mask_path {mask_path} --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
        )

    # Get the masked image of the object
    with Timer("Image Segmentation"):
        os.system(
            f"python ./data_process/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT {category} --output_path {base_path}/{case_name}/shape/masked_image.png"
        )

    with Timer("Shape Prior Generation"):
        os.system(
            f"python ./data_process/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
        )

if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        os.system(
            f"python ./data_process/dense_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_OTHER:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        os.system(
            f"python ./data_process/data_process_pcd.py --base_path {base_path} --case_name {case_name}"
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        os.system(
            f"python ./data_process/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        os.system(
            f"python ./data_process/data_process_track.py --base_path {base_path} --case_name {case_name}"
        )
