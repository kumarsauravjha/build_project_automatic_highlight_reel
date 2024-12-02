# %%
import cv2
import numpy as np
import csv
import pandas as pd

# %%
def create_transition(frame1, frame2, transition_type='fade', duration_frames=30):
    """Create a smooth transition between two frames."""
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)
    transition_frames = []
    for i in range(duration_frames):
        progress = i / (duration_frames - 1)
        if transition_type == 'fade':
            frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        else:
            raise ValueError("Unsupported transition type")
        transition_frames.append(frame.astype(np.uint8))
    return transition_frames

#%%
def apply_cartoon(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    gray = cv2.medianBlur(gray, 7)
    # Detect edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Smooth the original frame
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    # Combine edges and smoothed color
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

#%%
def apply_invert(frame):
    return cv2.bitwise_not(frame)

#%%
def apply_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# %%
def main(input_video, csv_file, output_video, transition_frames=30):
    """Main function to process the video and add transitions."""

    # Open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Parse CSV to determine which frames to process
    df = pd.read_csv(csv_file)
    filter_df = df[df['value'] == 1]
    frame_filter = filter_df.set_index('frame')['value'].to_dict()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_fps = int(fps*1.2)
    # Initialize video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (width, height))

    last_frame = None
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = apply_cartoon(frame)
        # Process only frames marked as live (1) in the CSV
        if frame_filter.get(frame_number, 0) == 1:
            # If there’s a gap, add a smooth transition
            if last_frame is not None and frame_number - 1 not in frame_filter:
             
                frame = apply_grayscale(frame)
                transition = create_transition(last_frame, frame, 'fade', transition_frames)
                for t_frame in transition:
                    out.write(t_frame)
                    # for _ in range(4):
                    #     out.write(t_frame)
            
            # Write the current frame
            out.write(frame)
            last_frame = frame

        frame_number += 1
        # # For testing purposes, break after a specific frame
        if frame_number == 6000:  # Remove or adjust as needed
            break

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video}")

# %%
# Input parameters
input_video = "video.mp4"
csv_file = "smoothed_predictions.csv"
output_video = "output_video11.mp4"
transition_frames = 30

# Call the main function
main(input_video, csv_file, output_video, transition_frames)

# %%
