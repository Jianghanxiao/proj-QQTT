import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image

"""
Hyper parameters
"""
TEXT_PROMPT = "toy."
# IMG_PATH = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/test_enhance_sdxl/single_lift_dinosor_enhanced.png"
# CASE_NAME = "test"
BASE_PATH = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reference_image/"
CASE_NAME = "0"
IMG_PATH = f"{BASE_PATH}/{CASE_NAME}.png"
SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reference_image"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

existDir(OUTPUT_DIR)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image_source, image = load_image(img_path)

sam2_predictor.set_image(image_source)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)


confidences = confidences.numpy().tolist()
class_names = labels

OBJECTS = class_names

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

for i in range(np.shape(masks)[0]):
    # masks is n * H * W
    Image.fromarray((masks[i] * 255).astype(np.uint8)).save(
        f"{OUTPUT_DIR}/{[i]}.png"
    )

if True:
    raw_img = cv2.imread(img_path)
    mask_img = cv2.imread(f"{OUTPUT_DIR}/[0].png", cv2.IMREAD_GRAYSCALE)

    ref_img = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = mask_img > 0
    ref_img[mask_bool, :3] = raw_img[mask_bool]
    ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255


    cv2.imwrite(f"{OUTPUT_DIR}/{CASE_NAME}.png", ref_img)


