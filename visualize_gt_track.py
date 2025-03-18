import os
import cv2
import numpy as np
import mediapy as media
import pickle

# Load original video
video_path = "data/different_types_gt_track/single_lift_dinosor/0.mp4"  # Adjust path as needed
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

video = media.read_video(video_path)
height, width = video.shape[1:3]
frame_len, height, width, _ = video.shape

# Load tracking results from .pkl file

pkl_filename = video_path.replace(".mp4", "_tracking.pkl")
if not os.path.exists(pkl_filename):
    raise FileNotFoundError(f"Tracking data '{pkl_filename}' not found.")

with open(pkl_filename, "rb") as f:
    all_pos = pickle.load(f)
print(f"Tracking data loaded: {pkl_filename}")
# Define colors for tracking points
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (128, 0, 128), (128, 128, 0), (0, 128, 128)]

output_video_path = video_path.replace(".mp4", "_tracking.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
fps = 30  # Set FPS (adjust as needed)
frame_size = (width, height)
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

replay = True
while replay:
    
    for i in range(video.shape[0]):
        frame = np.copy(video[i])  # Get frame from video
        for j, (x, y) in enumerate(all_pos[i]):
            cv2.circle(frame, (x, y), 5, colors[j], -1)  # Draw tracking points
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Tracking Visualization", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.write(frame_bgr)
        key = cv2.waitKey(50) & 0xFF

        if key == 13:  # 'Enter' key to exit
            replay = False
            print("Visualization stopped.")
            break
        
          

cv2.destroyAllWindows()

out.release()