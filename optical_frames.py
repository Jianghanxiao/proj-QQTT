
import os
import cv2
import numpy as np
import torch
import mediapy as media
import flow_vis
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import copy

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the RAFT optical flow model
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

# Load a local video file
video_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types/single_lift_cloth_1/color/0.mp4"  # Change this to your local video file
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

video = media.read_video(video_path)
# video = media.resize_video(video, (480, 768))  # Resize to a standard format
height, width = video.shape[1:3]

# Compute Optical Flow
optical_flows = []
for i in range(video.shape[0] - 1):
    image1 = video[i].astype(np.float32) / 127.5 - 1.0
    image1 = image1.transpose(2, 0, 1)[None]
    image2 = video[i + 1].astype(np.float32) / 127.5 - 1.0
    image2 = image2.transpose(2, 0, 1)[None]

    flow = model(torch.tensor(image1).to(device), torch.tensor(image2).to(device))
    flow = flow[-1][0].detach().cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy format
    optical_flows.append(flow)

optical_flows = np.stack(optical_flows)

# Release GPU Memory
del model
torch.cuda.empty_cache()

print("Optical flow computation complete!")


clicks = {}  # Stores clicked positions per frame
all_pos = np.zeros((video.shape[0], 2), dtype=int)
last_click = None
current_frame = 0  # Tracks which frame is currently displayed

def click_event(event, x, y, flags, param):
    """ Handles mouse clicks on frames to select tracking points. """
    global clicks, current_frame, last_click

    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click: Select a point to track
        clicks[current_frame] = (x, y)
        print(f"Selected: Frame {current_frame}, Position: ({x}, {y})")
        last_click = (current_frame, (x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click: Remove last selection
        if last_click:
            del clicks[last_click[0]]
            last_click = None
            print("Last selection removed.")

# Open a window to select tracking points
cv2.namedWindow("Select Points to Track")
cv2.setMouseCallback("Select Points to Track", click_event)

while True:
    img = cv2.cvtColor(video[current_frame], cv2.COLOR_RGB2BGR)

    # Draw selected points on the current frame
    for f, (x, y) in clicks.items():
        if f == current_frame:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Select Points to Track", img)
    key = cv2.waitKey(100) & 0xFF  # Wait for key press

    if key == ord('n'):  # Press 'n' to go to the next frame
        current_frame = min(current_frame + 1, video.shape[0] - 1)
    elif key == ord('p'):  # Press 'p' to go to the previous frame
        current_frame = max(current_frame - 1, 0)
    elif key == 13:  # Press 'Enter' to start tracking
        break

cv2.destroyAllWindows()

# ========== Track Points Over Frames ==========
def click2idx(click):
    """ Converts click coordinates to valid image indices """
    x, y = click
    return int(round(x)), int(round(y))

cur_pos = None
frames2 = []
tracked_positions = {}

# Track objects using optical flow
for i in range(video.shape[0]):
    if i in clicks:
        cur_pos = list(copy.copy(clicks[i]))
        tracked_positions[i] = cur_pos

    if cur_pos:
        x, y = click2idx(cur_pos)
        y = np.clip(y, 0, height - 2)
        x = np.clip(x, 0, width - 2)

        all_pos[i, 0] = x
        all_pos[i, 1] = y

        if i < optical_flows.shape[0]:
            cur_pos[0] += optical_flows[i, y, x, 0]
            cur_pos[1] += optical_flows[i, y, x, 1]

# ========== Visualization ==========
for i in range(video.shape[0]):
    fr = np.copy(video[i])
    x, y = all_pos[i]

    cv2.circle(fr, (x, y), 5, (0, 0, 255), -1)  # Draw red point
    cv2.imshow("Tracked Frames", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(50)  # Display each frame for 50 ms

cv2.destroyAllWindows()
print("Tracking visualization complete!")