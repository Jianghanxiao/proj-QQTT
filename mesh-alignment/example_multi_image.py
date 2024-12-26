import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
# images = [
#     Image.open("assets/example_multi_image/character_1.png"),
#     Image.open("assets/example_multi_image/character_2.png"),
#     Image.open("assets/example_multi_image/character_3.png"),
# ]


root_dir = '/home/haoyuyh3/Documents/maxhsu/qqtt/data-3dgs'
scene_name = 'rope_double_hand'
object_seg_id = 1

images = []
for cam_id in range(3):
    im = np.array(Image.open(os.path.join(root_dir, scene_name, 'color', str(cam_id), '0.png')))
    seg = np.array(Image.open(os.path.join(root_dir, scene_name, 'mask', str(cam_id), str(object_seg_id), '0.png'))).astype(np.float32)
    fg = np.zeros_like(im)
    mask_bool = seg > 0
    fg[mask_bool] = im[mask_bool]
    final_im = np.concatenate([fg.astype(np.uint8), np.expand_dims(seg, axis=-1).astype(np.uint8)], axis=-1)
    seg = np.expand_dims(seg, axis=-1)
    final_im = Image.fromarray(final_im)
    images.append(final_im)


# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

output_dir = os.path.join('./output', scene_name)
os.makedirs(output_dir, exist_ok=True)

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(os.path.join(output_dir, "sample_multi.mp4"), video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(os.path.join(output_dir, "sample.glb"))

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(os.path.join(output_dir, "sample.ply"))