from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd


import sys
sys.path.append("../")
from utils import get_bbox_center, get_bbox_width ,get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()
    
    
    def add_position_to_track(self,tracks):
        for object,object_track in tracks.items():
            for frame_num,track in enumerate(object_track):
                for track_id,track_data in track.items():
                    bbox=track_data['bbox']
                    
                    if object=='ball':
                        position=get_bbox_center(bbox)
                    else:
                        position=get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position']=position

    def interpollate_ball_postions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        #Interpolate missing values
        df_ball_positions=df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        #return in original format
        ball_positions=[{1:{'bbox':x}}for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        
    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+=detections_batch
            
        return detections
    
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        
        if read_from_stub==True and stub_path is not None and os.path.exists(stub_path):
            # Load tracks from stub if it exists
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
        
        detections=self.detect_frames(frames)
        
        tracks={
            "referee": [],
            "player": [],
            "ball": []
        }

        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}

            #convert to supervision format
            detections_supervision=sv.Detections.from_ultralytics(detection)
            
            
            #convert goalkeeper to player
            for obj_ind,class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id]=='goalkeeper':
                    detections_supervision.class_id[obj_ind]=cls_names_inv['player']

            
            #track objects
            detections_with_tracks = self.tracker.update_with_detections(detections_supervision)

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detections_with_tracks:
                bbox= frame_detection[0].tolist()
                class_id=frame_detection[3]
                track_id=frame_detection[4] 
                if cls_names[class_id]=='player':
                    tracks["player"][frame_num][track_id]={"bbox": bbox}
                
                if cls_names[class_id]=='referee':
                    tracks["referee"][frame_num][track_id]={"bbox": bbox}
                    
            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
            
            
        if stub_path is not None:
            # Save detections to stub if required
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            pass
        return tracks

    def draw_annotations(self,video_frames,tracks,team_ball_control,view_transformer=None,homographies=None,team_colours=None,team_passes=None):
        output_frames = []
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()
            
            player_dictionary=tracks["player"][frame_num]
            referee_dictionary=tracks["referee"][frame_num]
            ball_dictionary=tracks["ball"][frame_num]
            
            #Draw players
            for track_id,player in player_dictionary.items():
                colour=player.get('team_colour', (0, 0, 0))  
                frame=self.draw_ellipse(frame,player["bbox"],colour=colour ,track_id=track_id,track_info=player)
                
                if player.get('has_ball', False):
                    self.draw_triangle(frame,player["bbox"],colour=(0, 0, 255))

            #Draw referees
            for track_id,referee in referee_dictionary.items():
                frame=self.draw_ellipse(frame,referee["bbox"],colour=(0, 255, 255),track_id=track_id)
            
            
            #draw ball
            for track_id,ball in ball_dictionary.items():
                
                frame=self.draw_triangle(frame,ball["bbox"],colour=(255, 255, 0))
            
            if view_transformer is not None and homographies is not None:
                H = homographies[frame_num] if frame_num < len(homographies) else None
                
                # 1. Make it bigger (e.g., 1.5x the original size)
                # Original was 210x140. Let's try 315x210 (or go up to 420x280 for 2x)
                map_width, map_height = 315, 210 
                
                # 2. Calculate bottom-right position dynamically
                padding = 20
                frame_height, frame_width = frame.shape[:2]
                
                origin_x = frame_width - map_width - padding
                origin_y = frame_height - map_height - padding
                
                frame = view_transformer.draw_minimap(
                    frame, tracks, frame_num, H,
                    minimap_origin=(origin_x, origin_y),
                    minimap_size=(map_width, map_height),
                    team_colours=team_colours
                    )
                
            frame=self.draw_teams_control(frame,frame_num,team_ball_control,team_passes=team_passes)
            output_frames.append(frame)
        return output_frames
    
    
    
    def draw_triangle(self,frame,bbox,colour):
        y=int(bbox[1])
        x,_=get_bbox_center(bbox)
        
        triangle_points= np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        
        cv2.drawContours(frame, [triangle_points], 0, colour, thickness=cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, colour, thickness=cv2.FILLED)
        
        return frame

        
        
    def draw_ellipse(self,frame,bbox,colour,track_id=None,track_info=None):
        y2=int(bbox[3])
        
        x1_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            (x1_center, y2),
            axes=(int(width), int(width*0.35)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_AA
            
        )
        
        rectangle_width=40
        rectangle_height=20
        x1_rect=x1_center- rectangle_width//2
        x2_rect=x1_center+ rectangle_width//2
        y1_rect=(y2-rectangle_height//2)+15
        y2_rect=(y2+rectangle_height//2)+15
        
        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                colour,
                cv2.FILLED
            )
            x1_text=x1_rect+12
            if track_id>99:
                x1_text-=10
            
            cv2.putText(
                frame,
                str(track_id),
                (x1_text, y1_rect+15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0,0),
                thickness=2
            )
            if track_info is not None:
                y_text = int(y2 + 40) # Pushed down slightly to clear the ID box
            
                speed = track_info.get('speed')
                distance = track_info.get('distance')
            
                if speed is not None and distance is not None:
                    # Only draw if they are actually moving
                    if speed > 2.0: 
                        cv2.putText(
                            frame,
                            f"{speed:.1f} km/h",
                            (x1_center - 35, y_text), # Shifted left slightly to center it
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2
                        )
                    
                    # Draw total distance covered below the speed
                    cv2.putText(
                        frame,
                        f"{distance:.1f} m",
                        (x1_center - 30, y_text + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
        
        
        return frame
        
    def draw_teams_control(self, frame, frame_num, team_ball_control, team_passes=None):
        # 1. Get frame width to calculate the right side
        frame_width = frame.shape[1]
        
        # Draw semi-transparent rectangle at TOP-RIGHT corner
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame_width - 280, 0), (frame_width, 90), (50, 50, 50), -1) 
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        team1_num_frames = team_ball_control_till_frame.count(1)
        team2_num_frames = team_ball_control_till_frame.count(2)
        
        total_frames = team1_num_frames + team2_num_frames
        team1 = team1_num_frames / total_frames if total_frames > 0 else 0
        team2 = team2_num_frames / total_frames if total_frames > 0 else 0
        
        # Team 1 Stats (Shifted X coordinate to frame_width - 270)
        t1_passes = team_passes.get(1, 0) if team_passes else 0
        cv2.putText(
            frame,
            f'Team 1: {team1*100:.1f}%  | Passes: {t1_passes}',
            (frame_width - 270, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2
        )
        
        # Team 2 Stats (Shifted X coordinate to frame_width - 270)
        t2_passes = team_passes.get(2, 0) if team_passes else 0
        cv2.putText(
            frame,
            f'Team 2: {team2*100:.1f}%  | Passes: {t2_passes}',
            (frame_width - 270, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2
        )

        return frame