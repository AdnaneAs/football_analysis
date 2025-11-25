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

## Modules Used
The following modules are used in this project:
- YOLO v11
- BoT-SORT
- Kmeans
- Optical Flow
- Kalman Filter
- Perspective Transformation

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas