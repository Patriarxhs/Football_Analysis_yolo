from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np


import sys
sys.path.append("../")
from utils import get_bbox_center, get_bbox_width

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()
        

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

    def draw_annotations(self,video_frames,tracks):
        output_frames = []
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()
            
            player_dictionary=tracks["player"][frame_num]
            referee_dictionary=tracks["referee"][frame_num]
            ball_dictionary=tracks["ball"][frame_num]
            
            #Draw players
            for track_id,player in player_dictionary.items():
                colour=player.get('team_colour', (0, 255, 0))  
                frame=self.draw_ellipse(frame,player["bbox"],colour=colour ,track_id=track_id)

            #Draw referees
            for track_id,referee in referee_dictionary.items():
                frame=self.draw_ellipse(frame,referee["bbox"],colour=(0, 255, 255),track_id=track_id)
            
            
            #draw ball
            for track_id,ball in ball_dictionary.items():
                
                frame=self.draw_triangle(frame,ball["bbox"],colour=(255, 255, 0))
                
                
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

        
        
    def draw_ellipse(self,frame,bbox,colour,track_id=None):
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
        
        
        return frame
        