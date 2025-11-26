from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

from tqdm import tqdm

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        
        # Initialize Kalman Filter
        # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        
        interpolated_ball_positions = []
        
        print("Interpolating ball positions...")
        for bbox in tqdm(ball_positions):
            # Predict next state
            prediction = kf.predict()
            
            if bbox: # Measurement available
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
                kf.correct(measurement)
                interpolated_ball_positions.append(bbox)
            else:
                # Use prediction
                center_x = prediction[0][0]
                center_y = prediction[1][0]
                
                # Use last known width/height or default
                if interpolated_ball_positions and interpolated_ball_positions[-1]:
                    last_bbox = interpolated_ball_positions[-1]
                    width = last_bbox[2] - last_bbox[0]
                    height = last_bbox[3] - last_bbox[1]
                else:
                    width = 0
                    height = 0
                
                if width > 0:
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    interpolated_ball_positions.append([x1, y1, x2, y2])
                else:
                    interpolated_ball_positions.append([])

        # Backfill missing values at the start
        first_valid_bbox = None
        for bbox in interpolated_ball_positions:
            if bbox:
                first_valid_bbox = bbox
                break
        
        if first_valid_bbox:
            for i in range(len(interpolated_ball_positions)):
                if not interpolated_ball_positions[i]:
                    interpolated_ball_positions[i] = first_valid_bbox
                else:
                    break

        ball_positions = [{1: {"bbox":x}} for x in interpolated_ball_positions]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in tqdm(range(0,len(frames),batch_size), desc="Detecting frames"):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Use BoT-SORT tracker from Ultralytics
        print("Tracking objects...")
        # Note: model.track doesn't support tqdm directly on the internal loop easily without callbacks,
        # but it prints its own progress. We'll just print a start message.
        results = self.model.track(frames, persist=True, tracker="botsort.yaml")

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, result in enumerate(results):
            names = result.names

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names[cls_id]
                bbox = box.xyxy[0].tolist()
                
                # Track ID might be None if not tracked
                track_id = int(box.id[0]) if box.id is not None else None

                # Handle Goalkeeper as Player
                if class_name == "goalkeeper":
                    class_name = "player"
                
                if class_name == "player" and track_id is not None:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if class_name == "referee" and track_id is not None:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
                if class_name == "ball":
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        # Avoid Division by Zero
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        print("Drawing annotations...")
        for frame_num, frame in tqdm(enumerate(video_frames), total=len(video_frames)):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                if ball["bbox"]: # Check if bbox is not empty
                    frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames