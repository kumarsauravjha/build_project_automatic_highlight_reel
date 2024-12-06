# %%
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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
def add_text_for_duration(frame, text, frame_number, start_frame, end_frame, position, font_scale=1, color=(0, 255, 0), thickness=2):
    """Add text to a frame for a specific duration."""
    if start_frame <= frame_number <= end_frame:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_fps = int(fps * 1.2)
    # Initialize video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (width, height))

    last_frame = None
    frame_number = 0
    
    # Define text properties
    text = "Day 3, Match 7"
    start_frame = 0
    end_frame = int(fps * 2)  # Display text for 2 seconds
    text_position = (50, height // 2)  # Fixed position for text

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text for the first 2 seconds
        frame = add_text_for_duration(frame, text, frame_number, start_frame, end_frame, text_position)

        # Process only frames marked as live (1) in the CSV
        if frame_filter.get(frame_number, 0) == 1:
            # If there’s a gap, add a smooth transition
            if last_frame is not None and frame_number - 1 not in frame_filter:
                transition = create_transition(last_frame, frame, 'fade', transition_frames)
                for t_frame in transition:
                    out.write(t_frame)
            
            # Write the current frame
            out.write(frame)
            last_frame = frame

        frame_number += 1
        # Remove or adjust the following line as needed
        if frame_number == 8000:
            break

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video}")

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with transitions based on CSV input.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing frame information.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to the output video file.")
    parser.add_argument("--transition_frames", type=int, default=30, help="Number of frames to use for transitions.")

    args = parser.parse_args()

    main(args.input_video, args.csv_file, args.output_video, args.transition_frames)
