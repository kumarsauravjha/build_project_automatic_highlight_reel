
import cv2
import numpy as np
import csv

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

def main(input_video, csv_file, output_video, transition_frames=30):
    # Open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Parse CSV to determine which frames to process
    frame_filter = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_filter[int(row['frame'])] = int(row['value'])

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    last_frame = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process video with live frames and smooth transitions.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("csv_file", help="Path to the CSV file specifying live frames.")
    parser.add_argument("output_video", help="Path to save the output video.")
    args = parser.parse_args()

    main(args.input_video, args.csv_file, args.output_video)
