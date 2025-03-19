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
    output_dir = '/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_dynamic_gnn_out_domain'

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'output_dynamic_gnn_out_domain.txt')

    with open(log_file_path, 'w') as log_file:

        exp_names = sorted(os.listdir(output_dir))

        all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = [], [], [], []
        all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []

        scene_metrics = {}

        for exp_name in exp_names:

            tmp_exp_name = exp_name.replace('render-', '').replace('-model_50', '')
            target_name = tmp_exp_name.split("_to_")[1]

            scene_dir = os.path.join(root_data_dir, target_name)
            output_scene_dir = os.path.join(output_dir, exp_name)

            frame_len = len(os.listdir(os.path.join(scene_dir, 'color', '0')))
            assert frame_len == len(os.listdir(os.path.join(output_scene_dir, '0'))), "Number of frames do not match"
            test_f_idx_range = list(range(1, frame_len))  # ignore the first frame

            print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

            psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

            # for view_idx in range(3):
            for view_idx in range(1):   # only consider the first view

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

            scene_metrics[exp_name] = {
                'psnr_test': np.mean(psnrs_test),
                'ssim_test': np.mean(ssims_test),
                'lpips_test': np.mean(lpipss_test),
                'iou_test': np.mean(ious_test)
            }

            all_psnrs_test.extend(psnrs_test)
            all_ssims_test.extend(ssims_test)
            all_lpipss_test.extend(lpipss_test)
            all_ious_test.extend(ious_test)

            print(f'===== Exp: {exp_name} =====')
            print(f'\t PSNR (test): {np.mean(psnrs_test):.4f}')
            print(f'\t SSIM (test): {np.mean(ssims_test):.4f}')
            print(f'\t LPIPS (test): {np.mean(lpipss_test):.4f}')
            print(f'\t IoU (test): {np.mean(ious_test):.4f}')

        print('===== Overall Results Across All Scenes =====')
        print(f'\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}')
        print(f'\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}')
        print(f'\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}')
        print(f'\t Overall IoU (test): {np.mean(all_ious_test):.4f}')
        
        overall_psnr_test = np.mean(all_psnrs_test)
        overall_ssim_test = np.mean(all_ssims_test)
        overall_lpips_test = np.mean(all_lpipss_test)
        overall_iou_test = np.mean(all_ious_test)

        # Write overall metrics to log file
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("OVERALL RESULTS ACROSS ALL SCENES\n")
        log_file.write("=" * 80 + "\n\n")
        
        log_file.write(f"Overall PSNR (test): {overall_psnr_test:.6f}\n")
        log_file.write(f"Overall SSIM (test): {overall_ssim_test:.6f}\n")
        log_file.write(f"Overall LPIPS (test): {overall_lpips_test:.6f}\n")
        log_file.write(f"Overall IoU (test): {overall_iou_test:.6f}\n\n")
        
        # Create a compact table of all scene metrics
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("COMPACT METRICS TABLE BY SCENE\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Header
        log_file.write(f"{'Scene':<50} | ")
        log_file.write(f"{'PSNR-test':<12} | {'SSIM-test':<12} | {'LPIPS-test':<14} | {'IoU-test':<12}\n")
        log_file.write("-" * 160 + "\n")
        
        # Scene rows
        for scene in exp_names:
            metrics = scene_metrics[scene]
            log_file.write(f"{scene[:50]:<50} | ")            
            log_file.write(f"{metrics['psnr_test']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_test']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_test']:<14.6f} | ")
            log_file.write(f"{metrics['iou_test']:<12.6f}\n")
        
        # Overall row
        log_file.write("-" * 160 + "\n")
        log_file.write(f"{'OVERALL':<50} | ")        
        log_file.write(f"{overall_psnr_test:<12.6f} | ")
        log_file.write(f"{overall_ssim_test:<12.6f} | ")
        log_file.write(f"{overall_lpips_test:<14.6f} | ")
        log_file.write(f"{overall_iou_test:<12.6f}\n")
        
        print(f"\nMetrics have been saved to: {log_file_path}")


##### Ignore the first frame #####
# ===== Overall Results Across All Scenes =====
#          Overall PSNR (test): 28.5245
#          Overall SSIM (test): 0.9613
#          Overall LPIPS (test): 0.0334
#          Overall IoU (test): 0.6943