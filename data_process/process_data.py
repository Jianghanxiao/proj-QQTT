import os
from argparse import ArgumentParser
import time

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types",
)
parser.add_argument("--case_name", type=str)
# The category of the object used for segmentation
parser.add_argument("--category", type=str)
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

start = time.time()

if PROCESS_SHAPE_PRIOR:
    existDir(f"{base_path}/{case_name}/shape")
    # Get the high-resolution of the image to prepare for the trellis generation
    print(f"Image Upscaler: Processing {case_name}")
    os.system(
        f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
    )
    end = time.time()
    print(f"Time for Image Upscaler: {end - start}")
    start = end


if PROCESS_SEG_TRACK:
    # Get the masks of the controller and the object using GroundedSAM2
    print(f"Segmentation: Processing {case_name}")
    os.system(
        f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
    )
    print("Segmentation: Done")

if PROCESS_OTHER:
    pass
