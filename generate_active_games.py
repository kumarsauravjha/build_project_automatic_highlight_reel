import cv2
import pandas as pd
import argparse
from pathlib import Path

def load_frame_values(csv_path):
    """
    Load frame values from CSV file into a dictionary.
    If a frame doesn't exist in the CSV, it will return 0 by default.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['frame'], df['value']))

def process_active_frames(video_path, csv_path, output_path=None):
    """
    Process video file and save only frames where value == 1.
    """
    # Load frame values
    frame_values = load_frame_values(csv_path)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_number = 0  # Start at frame 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get value for current frame (default to 0 if not in CSV)
        value = frame_values.get(frame_number, 0)
        
        # Save only frames where value == 1
        if value == 1:
            text = f"Frame: {frame_number}, Value: {value}"
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)  # Write to output video
        
        frame_number += 1
    
    # Cleanup
    cap.release()
    out.release()
    print(f"Active game frames saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Overlay CSV values on video frames')
    parser.add_argument('video_path', type=Path, help='Path to input video file')
    parser.add_argument('csv_path', type=Path, help='Path to CSV file with frame values')
    parser.add_argument('--output', '-o', type=Path, help='Optional path for output video')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    process_active_frames(args.video_path, args.csv_path, args.output)

if __name__ == "__main__":
    main()
