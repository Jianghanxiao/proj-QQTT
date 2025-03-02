import os
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
from tqdm import tqdm
import torch
# import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import numpy as np


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


if __name__ == "__main__":
    
    root_data_dir = '/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian_data'
    output_dir = '/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_dynamic_fixed_relation'

    scene_name = sorted(os.listdir(root_data_dir))

    all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = [], [], [], []
    all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []

    for scene in scene_name:

        scene_dir = os.path.join(root_data_dir, scene)
        output_scene_dir = os.path.join(output_dir, scene)

        # Load frame split info
        with open(os.path.join(scene_dir, 'split.json'), 'r') as f:
            info = json.load(f)
        frame_len = info['frame_len']
        train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))   # +1 if ignoring the first frame
        test_f_idx_range = list(range(info["test"][0], info["test"][1]))

        print("train indices range from", train_f_idx_range[0], "to", train_f_idx_range[-1])
        print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

        psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
        psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

        # for view_idx in range(3):
        for view_idx in range(1):   # only consider the first view

            for frame_idx in train_f_idx_range:

                gt = np.array(Image.open(os.path.join(scene_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                gt_mask = np.array(Image.open(os.path.join(scene_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                gt_mask = gt_mask.astype(np.float32) / 255.

                render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                human_mask = np.array(Image.open(os.path.join(scene_dir, 'human_mask', str(view_idx), '0', f'{frame_idx}.png')))
                inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                gt = gt.astype(np.float32) * gt_mask[..., None]
                render = render[:, :, :3].astype(np.float32)

                gt = gt * inv_human_mask[..., None]
                render = render * inv_human_mask[..., None]
                render_mask = render_mask * inv_human_mask

                gt_tensor = img2tensor(gt)
                render_tensor = img2tensor(render)

                psnrs_train.append(psnr(render_tensor, gt_tensor).item())
                ssims_train.append(ssim(render_tensor, gt_tensor).item())
                lpipss_train.append(lpips(render_tensor, gt_tensor).item())
                ious_train.append(compute_iou(gt_mask > 0, render_mask > 0))

            for frame_idx in test_f_idx_range:
                    
                gt = np.array(Image.open(os.path.join(scene_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                gt_mask = np.array(Image.open(os.path.join(scene_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                gt_mask = gt_mask.astype(np.float32) / 255.

                render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                human_mask = np.array(Image.open(os.path.join(scene_dir, 'human_mask', str(view_idx), '0', f'{frame_idx}.png')))
                inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                gt = gt.astype(np.float32) * gt_mask[..., None]
                render = render[:, :, :3].astype(np.float32)

                gt = gt * inv_human_mask[..., None]
                render = render * inv_human_mask[..., None]
                render_mask = render_mask * inv_human_mask

                gt_tensor = img2tensor(gt)
                render_tensor = img2tensor(render)

                psnrs_test.append(psnr(render_tensor, gt_tensor).item())
                ssims_test.append(ssim(render_tensor, gt_tensor).item())
                lpipss_test.append(lpips(render_tensor, gt_tensor).item())
                ious_test.append(compute_iou(gt_mask > 0, render_mask > 0))

        all_psnrs_train.extend(psnrs_train)
        all_ssims_train.extend(ssims_train)
        all_lpipss_train.extend(lpipss_train)
        all_ious_train.extend(ious_train)

        all_psnrs_test.extend(psnrs_test)
        all_ssims_test.extend(ssims_test)
        all_lpipss_test.extend(lpipss_test)
        all_ious_test.extend(ious_test)

        print(f'===== Scene: {scene} =====')
        print(f'\t PSNR (train): {np.mean(psnrs_train):.4f}')
        print(f'\t SSIM (train): {np.mean(ssims_train):.4f}')
        print(f'\t LPIPS (train): {np.mean(lpipss_train):.4f}')
        print(f'\t IoU (train): {np.mean(ious_train):.4f}')

        print(f'\t PSNR (test): {np.mean(psnrs_test):.4f}')
        print(f'\t SSIM (test): {np.mean(ssims_test):.4f}')
        print(f'\t LPIPS (test): {np.mean(lpipss_test):.4f}')
        print(f'\t IoU (test): {np.mean(ious_test):.4f}')

    print('===== Overall Results Across All Scenes =====')
    print(f'\t Overall PSNR (train): {np.mean(all_psnrs_train):.4f}')
    print(f'\t Overall SSIM (train): {np.mean(all_ssims_train):.4f}')
    print(f'\t Overall LPIPS (train): {np.mean(all_lpipss_train):.4f}')
    print(f'\t Overall IoU (train): {np.mean(all_ious_train):.4f}')

    print(f'\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}')
    print(f'\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}')
    print(f'\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}')
    print(f'\t Overall IoU (test): {np.mean(all_ious_test):.4f}')


##### Include the first frame #####
# ===== Overall Results Across All Scenes =====
#          Overall PSNR (train): 30.0737
#          Overall SSIM (train): 0.9646
#          Overall LPIPS (train): 0.0251
#          Overall IoU (train): 0.7936
#          Overall PSNR (test): 28.0637
#          Overall SSIM (test): 0.9615
#          Overall LPIPS (test): 0.0382
#          Overall IoU (test): 0.6881


##### Ignore the first frame #####
# ===== Overall Results Across All Scenes =====
#          Overall PSNR (train): 30.0279
#          Overall SSIM (train): 0.9645
#          Overall LPIPS (train): 0.0252
#          Overall IoU (train): 0.7928
#          Overall PSNR (test): 28.0637
#          Overall SSIM (test): 0.9615
#          Overall LPIPS (test): 0.0382
#          Overall IoU (test): 0.6881