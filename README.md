# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO v11, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Key Features
- **YOLO v11**: State-of-the-art AI object detection model.
- **BoT-SORT Tracking**: Robust tracking with Re-ID to reduce ID switching.
- **Camera Smoothing**: Stabilizes camera movement estimation using moving averages.
- **Smart Team Assignment**: Uses a voting mechanism to prevent team color flickering.
- **Kalman Filter Interpolation**: Physics-based ball trajectory prediction for smoother tracking.
- **Perspective Transformation**: Represents scene depth and perspective.
- **Speed & Distance**: Calculates player metrics in real-world units.
- **Tactical Analysis**: Generates Voronoi control maps and player heatmaps to visualize space control and movement density.

## Modules Used
The following modules are used in this project:
- YOLO v11
- BoT-SORT
- Kmeans
- Optical Flow
- Kalman Filter
- Perspective Transformation
- Tactical Analysis (Voronoi & Heatmaps)

## Requirements
To run this project, you need to have the following requirements installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Scipy
- Seaborn

## Usage
1. Place your input video in the `input_videos` folder.
2. Update the `video_path` in `main.py` if necessary.
3. Run the main script:
   ```bash
   python main.py
   ```
4. The results will be saved in the `output_videos` folder, organized by video name and timestamp. Each run produces:
   - `output_video.avi`: The annotated video with tracking, speed, and distance.
   - `voronoi_analysis.avi`: A video showing territorial control (Voronoi diagrams).
   - `heatmap_player_X.png`: A heatmap of the most active player's movement.
   - Stub files for tracking and camera movement (to speed up subsequent runs).