import cv2
import numpy as np
import argparse
import csv

def process_frame(frame, apply_filters=True, crop=None):
    if crop:
        h, w = frame.shape[:2]
        x0, y0, crop_w, crop_h = [int(c * 0.01 * dim) for c, dim in zip(crop, [w, h, w, h])]
        frame = frame[y0:y0+crop_h, x0:x0+crop_w]

    if apply_filters:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to BGR for colored edge overlay
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine original frame with colored edges
        result = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)
        
        return result
    else:
        return frame

def crossfade_transition(frame1, frame2, steps=10):
    """Generate intermediate frames for smooth transitions."""
    blended_frames = []
    for alpha in np.linspace(0, 1, steps):
        blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        blended_frames.append(blended_frame)
    return blended_frames

def main():
    parser = argparse.ArgumentParser(description='Process video with optional CSV frame filtering')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--csv', type=str, help='Path to CSV file for frame filtering')
    parser.add_argument('--filters', action='store_true', help='Apply video effects')
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X0', 'Y0', 'W', 'H'),
                        help='Crop video as percentage: x0 y0 width height')
    args = parser.parse_args()

    input_video = cv2.VideoCapture(args.input_video)
    if not input_video.isOpened():
        print("Error opening video file")
        return

    original_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    if args.crop:
        x0, y0, crop_w, crop_h = [int(c * 0.01 * dim) for c, dim in zip(args.crop, [original_width, original_height, original_width, original_height])]
        width, height = crop_w, crop_h
    else:
        width, height = original_width, original_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    frame_filter = {}
    if args.csv:
        with open(args.csv, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                frame_filter[int(row['frame'])] = int(row['value'])

        frame_number = next((frame for frame, value in frame_filter.items() if value != 0), 0)
        input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    else:
        frame_number = 0

    last_frame = None  # Store the last processed frame for transitions

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        if not frame_filter or frame_filter.get(frame_number, 0) == 1:
            # Process the frame
            processed_frame = process_frame(frame, args.filters, args.crop)

            # If there's a gap, apply crossfade transition
            if last_frame is not None and frame_number not in frame_filter.keys():
                crossfade_frames = crossfade_transition(last_frame, processed_frame)
                for fade_frame in crossfade_frames:
                    output_video.write(fade_frame)
                    cv2.imshow('Processed Video', fade_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Write the processed frame
            output_video.write(processed_frame)
            cv2.imshow('Processed Video', processed_frame)

            # Update last processed frame
            last_frame = processed_frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
