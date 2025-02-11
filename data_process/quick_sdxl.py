import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# let's download an  image
low_res_img = Image.open("/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types/single_lift_dinosor/color/0/0.png").convert("RGB")

prompt = "Hand manipulates a stuffed animal."

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("test.png")