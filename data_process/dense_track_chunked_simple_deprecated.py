# Use co-tracker to track the object and controller in the video with improved chunked processing
# This version uses sequential chunking without overlap

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import glob
import cv2
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--chunk_size", type=int, default=100, help="Number of frames per chunk")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
chunk_size = args.chunk_size

num_cam = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == num_cam
device = "cuda"


def read_mask(mask_path):
    """Convert the white mask into binary mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def exist_dir(dir):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_query_pixels_from_mask(mask_paths):
    """Get query pixels from mask files"""
    mask = None
    for mask_path in mask_paths:
        current_mask = read_mask(mask_path)
        if mask is None:
            mask = current_mask
        else:
            mask = np.logical_or(mask, current_mask)

    # Get query pixels from mask
    query_pixels = np.argwhere(mask)
    # Revert x and y
    query_pixels = query_pixels[:, ::-1]
    query_pixels = np.concatenate(
        [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
    )
    query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
    # Randomly select 5000 query points
    if query_pixels.shape[0] > 5000:
        query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]
    
    return query_pixels


def track_chunk_sequential(video, start_frame, end_frame, query_pixels):
    """
    Track a chunk using sequential processing
    
    Args:
        video: Full video tensor (B, T, C, H, W)
        start_frame: Start frame index in original video
        end_frame: End frame index in original video
        query_pixels: Query pixels (N, 3) - either from first frame or previous chunk
    
    Returns:
        Tuple of (tracks, visibility) for this chunk
    """
    print(f"Tracking chunk: frames {start_frame}-{end_frame}")
    
    # Load only the frames we need for this chunk
    chunk_video = video[:, start_frame:end_frame]
    
    # Initialize CoTracker for this chunk
    cotracker = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_online"
    ).to(device)
    
    # Initialize with first frame of chunk
    cotracker(video_chunk=chunk_video, is_first_step=True, queries=query_pixels[None])
    
    
    for ind in range(0, chunk_video.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, pred_visibility = cotracker(
            video_chunk=chunk_video[:, ind : ind + cotracker.step * 2]
        )
    return pred_tracks[0], pred_visibility[0]


def process_video_sequential_chunks(video, query_pixels, chunk_size, save_dir, camera_id):
    """
    Process video using sequential chunking without overlap
    
    Args:
        video: Full video tensor (B, T, C, H, W)
        query_pixels: Query pixels from first frame (N, 3)
        chunk_size: Number of frames per chunk
        save_dir: Directory to save visualizations
        camera_id: Camera ID for saving files
    
    Returns:
        Merged tracking results for the full video
    """
    total_frames = video.shape[1]
    num_points = query_pixels.shape[0]
    
    # Initialize output tensors
    all_tracks = torch.zeros((total_frames, num_points, 2), device=device)
    all_visibility = torch.zeros((total_frames, num_points), device=device, dtype=torch.bool)
    
    # Split video into non-overlapping chunks
    chunks = []
    start_idx = 0
    while start_idx < total_frames:
        end_idx = min(start_idx + chunk_size, total_frames)
        chunks.append((start_idx, end_idx))
        if end_idx == total_frames:
            break
        start_idx = end_idx
    
    print(f"Split video into {len(chunks)} chunks")
    
    # Process each chunk sequentially
    for i, (start_idx, end_idx) in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: frames {start_idx}-{end_idx}")
        
        # Use original query pixels for first chunk, or last frame positions for subsequent chunks
        if i == 0:
            # First chunk: use original query pixels
            current_query_pixels = query_pixels
            valid_mask = torch.ones_like(query_pixels[:, 0], device=device, dtype=torch.bool)
        else:
            # Subsequent chunks: use last frame positions from previous chunk
            prev_end_idx = chunks[i-1][1]
            last_frame_tracks = all_tracks[prev_end_idx - 1]  # Last frame of previous chunk
            last_frame_visibility = all_visibility[prev_end_idx - 1]
            
            # Get valid points from last frame
            valid_mask = last_frame_visibility > 0
            
            valid_tracks = last_frame_tracks[valid_mask]
            # Convert to query format (frame_idx, x, y)
            current_query_pixels = torch.zeros((valid_tracks.shape[0], 3), device=device)
            current_query_pixels[:, 1:] = valid_tracks
            print(f"Using {valid_tracks.shape[0]} valid points from previous chunk")
        
        # Track this chunk
        chunk_tracks, chunk_visibility = track_chunk_sequential(
            video, start_idx, end_idx, current_query_pixels
        )
        
        # Store results for this chunk
        all_tracks[start_idx:end_idx, valid_mask] = chunk_tracks
        all_visibility[start_idx:end_idx, valid_mask] = chunk_visibility
    
    return all_tracks[None], all_visibility[None]  # Add batch dimension


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in range(num_cam):
        print(f"Processing {i}th camera")
        
        # Load the video
        frames = iio.imread(f"{base_path}/{case_name}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        
        print(f"Video shape: {video.shape}")
        
        # Load the first-frame mask to get all query points from all masks
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        query_pixels = get_query_pixels_from_mask(mask_paths)
        
        print(f"Selected {query_pixels.shape[0]} query points")

        # Process video with sequential chunking approach
        pred_tracks, pred_visibility = process_video_sequential_chunks(
            video, query_pixels, chunk_size, 
            f"{base_path}/{case_name}/cotracker", i
        )
        
        # Visualize final results
        print("Visualizing final tracking results...")
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
        
        # Save the tracking data into npz (same format as original)
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
        visibility_to_save = pred_visibility[0].cpu().numpy()
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )
        
        print(f"Saved tracking results for camera {i}") 