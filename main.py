from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import os
from datetime import datetime
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from tactical_analysis import TacticalAnalysis
from tqdm import tqdm


def main():
    # Read Video
    video_path = 'input_videos/1126.mp4'
    video_frames = read_video(video_path)

    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_videos/{video_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=os.path.join(output_dir, 'track_stubs.pkl'))
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path=os.path.join(output_dir, 'camera_movement_stub.pkl'))
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    # Voting mechanism
    player_team_history = {}
    
    print("Assigning teams...")
    for frame_num, player_track in tqdm(enumerate(tracks['players']), total=len(tracks['players'])):
        for player_id, track in player_track.items():
            # Use predict_team to get fresh prediction
            team = team_assigner.predict_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            
            if player_id not in player_team_history:
                player_team_history[player_id] = []
            
            player_team_history[player_id].append(team)
            
            # Vote
            if len(player_team_history[player_id]) > 30: # Keep last 30 frames
                 player_team_history[player_id] = player_team_history[player_id][-30:]

            from collections import Counter
            most_common_team = Counter(player_team_history[player_id]).most_common(1)[0][0]
            
            tracks['players'][frame_num][player_id]['team'] = most_common_team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[most_common_team]

    
    # Assign Ball Aquisition
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    print("Assigning ball control...")
    for frame_num, player_track in tqdm(enumerate(tracks['players']), total=len(tracks['players'])):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        
        # Check if ball_bbox is valid
        if not ball_bbox:
            assigned_player = -1
        else:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If list is not empty, append the last team
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                # If list is empty (start of video), append a default value (e.g., None or 0)
                # Or simply do nothing if you handle empty lists later, but appending None keeps length consistent
                team_ball_control.append(None) 

    # Convert to numpy array, handling None values if necessary
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Tactical Analysis
    tactical_analysis = TacticalAnalysis()
    
    # 1. Voronoi Video
    voronoi_frames = tactical_analysis.draw_voronoi(tracks)
    save_video(voronoi_frames, os.path.join(output_dir, 'voronoi_analysis.avi'))
    
    # 2. Heatmap for the player with most movement (or a specific ID)
    # Find player with most frames tracked
    player_frame_counts = {}
    for frame in tracks['players']:
        for pid in frame.keys():
            player_frame_counts[pid] = player_frame_counts.get(pid, 0) + 1
            
    if player_frame_counts:
        most_active_player = max(player_frame_counts, key=player_frame_counts.get)
        heatmap = tactical_analysis.draw_heatmap(tracks, most_active_player)
        if heatmap is not None:
            cv2.imwrite(os.path.join(output_dir, f'heatmap_player_{most_active_player}.png'), heatmap)

    # Save video
    save_video(output_video_frames, os.path.join(output_dir, 'output_video.avi'))

if __name__ == '__main__':
    main()