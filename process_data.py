import os
from argparse import ArgumentParser
import time

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
PROCESS_SHAPE_PRIOR = True
PROCESS_SEG_TRACK = False
PROCESS_OTHER = False

base_path = args.base_path
case_name = args.case_name
category = args.category
TEXT_PROMPT = f"{category}.hand"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        print(f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SHAPE_PRIOR:
    existDir(f"{base_path}/{case_name}/shape")
    # # Get the high-resolution of the image to prepare for the trellis generation
    # with Timer("Image Upscale"):
    #     os.system(
    #         f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
    #     )

    # # Get the masked image of the object
    # with Timer("Image Segmentation"):
    #     os.system(
    #         f"python ./data_process/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT {category} --output_path {base_path}/{case_name}/shape/masked_image.png"
    #     )

    with Timer("Shape Prior Generation"):
        os.system(
            f"python ./data_process/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
        )

if PROCESS_SEG_TRACK:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        os.system(
            f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
        )

if PROCESS_OTHER:
    pass
