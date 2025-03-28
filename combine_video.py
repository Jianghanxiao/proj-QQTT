import os
import cv2

import cv2
import numpy as np
import os
# Base directories
base_dir = "./"
data_dir = os.path.join(base_dir, "data/different_types")
render_dir = os.path.join(base_dir, "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/exp_results/indomain_our/output_dynamic")
# render_dir = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/experiments"
combine_dir = os.path.join(base_dir, "combine")
print(data_dir, render_dir, combine_dir)
# Ensure combine directory exists
os.makedirs(combine_dir, exist_ok=True)
# Iterate through case directories in both data and render
for case_name in os.listdir(data_dir):
    data_case_path = os.path.join(data_dir, case_name, "color", "0.mp4")
    render_case_path = os.path.join(render_dir, case_name, "0_integrate.mp4")
    combine_case_path = os.path.join(combine_dir, case_name)
    print(f"Processing case: {case_name}")
    print(f"Data video: {data_case_path}")
    print(f"Render video: {render_case_path}")
    print(f"Output directory: {combine_case_path}")
    # Ensure both corresponding files exist
    if not os.path.exists(data_case_path) or not os.path.exists(render_case_path):
        print(f"Skipping {case_name}: Missing files.")
        continue
    # Ensure output directory exists for this case
    os.makedirs(combine_case_path, exist_ok=True)
    output_video_path = os.path.join(combine_case_path, "0_combine.mp4")
    # Open video files
    cap_origin = cv2.VideoCapture(data_case_path)
    cap_integrate = cv2.VideoCapture(render_case_path)
    # Get properties from integrate video (assuming it defines final resolution & FPS)
    frame_width = int(cap_integrate.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_integrate.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_integrate.get(cv2.CAP_PROP_FPS)
    # Crop dimensions
    x_start, x_end = 0, 8  # Crop range for both videos
    crop_width = x_end - x_start  # Final width of each cropped video
    # Define output video writer (combined width)
    combined_width = crop_width * 2  # Since we put both videos side by side
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width*2, frame_height))
    # Text settings (Fixed size based on "Reconstruction and Resimulation")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fixed_font_scale = 0.8  # Fixed text size for all texts
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    outline_color = (0, 0, 0)  # Black outline
    text_y = frame_height - 30  # Bottom position for text
    # Function to draw text with black outline
    def draw_text_with_outline(image, text, x, y, font, scale, color, thickness, outline_color):
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:  # Create black outline effect
            cv2.putText(image, text, (x + dx, y + dy), font, scale, outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    # Process both videos simultaneously
    while cap_origin.isOpened() and cap_integrate.isOpened():
        ret1, frame1 = cap_origin.read()
        ret2, frame2 = cap_integrate.read()
        if not ret1 or not ret2:
            break
        # Crop the x-axis section (keep full height)
        # cropped_origin = frame1[:, x_start:x_end]
        # cropped_integrate = frame2[:, x_start:x_end]
        cropped_origin = frame1
        cropped_integrate = frame2
        # Adjust brightness & contrast
        brightness_factor = 10
        contrast_factor = 1.1
        adjusted_origin = cv2.convertScaleAbs(cropped_origin, alpha=contrast_factor, beta=brightness_factor)
        adjusted_integrate = cv2.convertScaleAbs(cropped_integrate, alpha=contrast_factor, beta=brightness_factor)
        # Combine the two cropped videos side by side
        combined_frame = np.hstack((adjusted_origin, adjusted_integrate))
        # Determine text for integrate video
        frame_count = int(cap_integrate.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame index
        switch_frame = int(cap_integrate.get(cv2.CAP_PROP_FRAME_COUNT) * 0.7)  # 70% switch point
        # if frame_count < switch_frame:
        #     integrate_text = "Reconstruction and Resimulation"
        # else:
        #     integrate_text = "Future Prediction"
        # # **Keep all text the same size**
        # text1 = "Observation"
        # text2 = integrate_text
        # # Get text sizes at fixed scale
        # text1_size = cv2.getTextSize(text1, font, fixed_font_scale, font_thickness)[0]
        # text2_size = cv2.getTextSize(text2, font, fixed_font_scale, font_thickness)[0]
        # # **Ensure text fits within half width**
        # max_text_width = crop_width - 40  # Leave margin
        # if text2_size[0] > max_text_width:  # If text is too wide, reduce scale slightly
        #     while text2_size[0] > max_text_width and fixed_font_scale > 0.5:
        #         fixed_font_scale -= 0.05
        #         text2_size = cv2.getTextSize(text2, font, fixed_font_scale, font_thickness)[0]
        # # # Text X positions (aligned inside each half)
        # text1_x = (crop_width - text1_size[0]) // 2
        # text2_x = crop_width + (crop_width - text2_size[0]) // 2  # Centered in the right half
        # # Draw both texts with black border
        # draw_text_with_outline(combined_frame, text1, text1_x, text_y, font, fixed_font_scale, text_color, font_thickness, outline_color)
        # draw_text_with_outline(combined_frame, text2, text2_x, text_y, font, fixed_font_scale, text_color, font_thickness, outline_color)
        # Write frame
        out.write(combined_frame)
    # Release resources
    cap_origin.release()
    cap_integrate.release()
    out.release()
    print(f"Successfully processed: {case_name}")
cv2.destroyAllWindows()