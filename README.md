# **Build Project: Automatic Highlight Reel**

This repository contains a comprehensive solution for creating **automatic highlight reels** from video footage. It includes modular and well-structured code for video processing, filtering, animation, and data analysis. The project is designed for readability, extensibility, and efficient performance.


[![Watch the Demo Video](Screenshot%202024-12-04%20041631.jpg)](https://drive.google.com/file/d/11Wp242goVy7uF3vQu5haH9_fiUbHLrXN/view?usp=sharing)  
*Click on the image above to watch the demo video.*

[![Another Demo with effects]](https://drive.google.com/file/d/11Wp242goVy7uF3vQu5haH9_fiUbHLrXN/view?usp=sharing)  
---

Below is an example section you can add to your README that provides clear, improved instructions for recreating the final output reel. It integrates the repository’s images (adjust paths as needed) and offers a step-by-step guide:

---

## Recreating the Final Highlight Reel

This section walks you through the steps required to generate the final automatic highlight reel from your video footage, using the provided data and prediction scripts.

### Prerequisites

1. **Data Files:** Ensure you have `provided_data.csv` and `target.csv` placed in the `Data` directory.
2. **Video Footage:** Have your source video ready and note its file path.
   
### Step-by-Step Process

1. **Run the Time Classification Script**  
   Use `time_classification.py` to analyze the provided data and generate prediction files (/Data/predictions_all_new.csv):
   ```bash
   python3 Code/time_classification.py
   ```
   
   Upon completion, you should have an updated `predictions_all_new.csv` in the `Data` directory.

2. **Create the Final Highlight Reel**  
   With `predictions_all_new.csv` ready, you can now produce the highlight reel. Run `openCV_create_animation2.py` and provide it with the path to your video and predictions file:
   ```bash
   python3 Code/openCV_create_animation2.py --input_video path/to/your/video.mp4 --csv_file Data/predictions_all_new.csv --output_video pah_to_save_output_video
   ```
   
   This script will integrate the predictive data into a sample output highlight reel, emphasizing key moments from your footage.

### Final Output

After the scripts have finished running, you’ll find the generated highlight reel ready for review. This final video integrates the processed predictive data to create a polished, viewer-friendly highlight compilation automatically.

---

*By following the steps above, you’ll quickly and easily recreate the final output reel, benefiting from the project’s automated highlight detection and editing capabilities.*

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Code Details](#code-details)
- [Contributing](#contributing)

---

## **Overview**
The **Automatic Highlight Reel** is a downstream project that automates the identification of key moments in video footage, applying animations, and generating engaging highlight videos. This is especially useful for sports, events, or any scenario requiring quick post-processing of large video datasets.

Key components include:
- Time classification for event detection.
- Frame filtering for extracting significant frames.
- Video enhancement with OpenCV.
- Animated transitions and text overlays.

---

## **Features**
- **Event-Based Frame Filtering:** Automatically identifies and extracts relevant moments.
- **Custom Animations:** Add transitions, floating text, and effects to enhance video quality.
- **OpenCV Integration:** For video processing and frame manipulation.
- **Modular Code:** Easy to extend and adapt to different use cases.
- **Data Analysis Support:** Includes scripts for analyzing and visualizing time-classified events.

---

## **Folder Structure**

```
build_project_automatic_highlight_reel/
│
├── Code/
│   ├── animation.py                  # Core animation functions for video transitions.
│   ├── animation_new.py              # Extended animation functionalities.
│   ├── data_analysis.py              # Scripts for analyzing filtered frames and time data.
│   ├── filter_predictions.py         # Logic to filter frames based on predictions.
│   ├── openCV_create_animation.py    # Main OpenCV-based video creation script.
│   ├── time_classification.py        # Time-based event classification logic.
│   ├── time_classification_modeling_on_all_data.py
│                                     # Model training for time classification.
│   ├── time_classification_with_results.ipynb
│                                     # Notebook showcasing results and analysis.
│
└── Data/                             # (Optional) Dataset for testing and modeling (not included).
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/build_project_automatic_highlight_reel.git
   cd build_project_automatic_highlight_reel
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. **Frame Filtering:**
   Use `filter_predictions.py` to process raw predictions and filter significant frames.

2. **Video Processing with Animations:**
   Execute `openCV_create_animation.py` to create a video with transitions, animations, and overlays.

3. **Data Analysis:**
   Run `data_analysis.py` or explore `time_classification_with_results.ipynb` to analyze the performance of the classification model.

4. **Training Time Classification Model:**
   Use `time_classification_modeling_on_all_data.py` to train and test the time classification model on your dataset.

---

## **Code Details**

### **Core Scripts**
- **`animation.py`:** Contains functions for creating smooth video transitions.
- **`openCV_create_animation.py`:** The main driver script for video processing, integrating animations, filtering, and writing output.

### **Classification Scripts**
- **`time_classification.py`:** Classifies time intervals to identify key moments.
- **`time_classification_modeling_on_all_data.py`:** Builds and evaluates a time classification model.

### **Data Analysis**
- **`data_analysis.py`:** Visualizes and analyzes filtered frames and model performance.
- **`time_classification_with_results.ipynb`:** Jupyter notebook showcasing model results.

---

## **Contributing**

Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature-name"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

