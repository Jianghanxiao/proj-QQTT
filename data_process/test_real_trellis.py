import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
import numpy as np
import json
from argparse import ArgumentParser
import pickle
import torch

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reference_image"
parser = ArgumentParser()
parser.add_argument("--case_name", type=str, default="0")
args = parser.parse_args()
case_name = args.case_name
output_dir = f"{base_path}/models/{case_name}"

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

exist_dir(f"{base_path}/models")
exist_dir(output_dir)


final_im = Image.open(f"{base_path}/{case_name}.png").convert("RGBA")
assert not np.all(np.array(final_im)[:, :, 3]==255)

# Run the pipeline
outputs = pipeline.run(
    final_im,
    # seed=0,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 1,
    # #     "cfg_strength": 7.5,
    # },
    # custom_coords=final_coords,
)

# exist_dir(output_dir)

video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
video = [
    np.concatenate([frame_gs, frame_mesh], axis=1)
    for frame_gs, frame_mesh in zip(video_gs, video_mesh)
]
imageio.mimsave(f"{output_dir}/visualization.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)
glb.export(f"{output_dir}/object.glb")

# Save Gaussians as PLY files
outputs["gaussian"][0].save_ply(f"{output_dir}/object.ply")
