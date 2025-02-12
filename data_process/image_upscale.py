from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_path", type=str)
parser.add_argument("--category", type=str)
args = parser.parse_args()

img_path = args.img_path
output_path = args.output_path
category = args.category


# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
low_res_img = Image.open(img_path).convert("RGB")

prompt = f"A human manipulates a {category} with hand."

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save(output_path)
