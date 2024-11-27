# Video Processing with Smooth Transitions
!(readme_image.webp)
This project processes a video to extract and incorporate only "live" frames (as specified in a CSV file) while applying smooth transitions between frames with hard cuts. The transitions are implemented using fade effects to ensure seamless playback.

Features
Frame Filtering: Processes frames marked as "live" (value = 1) in the input CSV file.
Smooth Transitions: Generates smooth fade transitions between non-consecutive "live" frames.
Customizable Output: Saves the processed video with transitions to a specified output file.
How It Works
Parses a CSV file where each row specifies the frame number and whether it should be included in the final output.
Detects hard cuts between non-consecutive frames and applies fade transitions to bridge the gap.
Outputs a new video file containing only the selected frames and transitions.
Usage
Run the script as follows:

bash
Copy code
python process_video.py input_video.mp4 frames.csv output_video.mp4
Inputs
input_video.mp4: Path to the input video file.
frames.csv: CSV file specifying frame numbers and their values (1 for live frames, 0 for skipped frames).
output_video.mp4: Path to save the processed video.
