import os
import pickle
import cv2
import numpy as np
import sys

sys.path.append('../')
from utils import calculate_distance

class CameraMovementEstimator:
    def __init__(self,frame):
        self.min_distance=5
        
        first_frame_grayscale=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features=np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20]=1
        mask_features[:,900:1050]=1
        
        
        self.lk_params=dict(winSize=(15,15),
                            maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        
        self.features=dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )
        
        
    def adjust_position(self, tracks,camera_movement_per_frame):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_data in track.items():
                    position=track_data['position']
                    camera_movement=camera_movement_per_frame[frame_num]
                    position_adjusted=(position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted']=position_adjusted
        return tracks
    
    def get_camera_movement(self,frames,read_stub=False,stub_path=None):
        #read from stub
        if read_stub==True and stub_path is not None and os.path.exists(stub_path):
            # Load tracks from stub if it exists
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
               
        
        camera_movement=[[0,0]]*len(frames) #x,y movement for each frame
        
        previous_frame_gray=cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        previous_features=cv2.goodFeaturesToTrack(previous_frame_gray, **self.features)
        
        for frame_num in range(1,len(frames)):
            frame_gray=cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features , _ , _ = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, previous_features, None,**self.lk_params)
            
            max_distance=0
            movement_x=0
            movement_y=0
            
            for i,(new,old) in enumerate(zip(new_features,previous_features))       :
                new_features_point= new.ravel()
                previous_features_point= old.ravel()
                
                distance=calculate_distance(new_features_point,previous_features_point)
                if distance>max_distance:
                    max_distance=distance
                    movement_x=new_features_point[0]-previous_features_point[0]
                    movement_y=new_features_point[1]-previous_features_point[1]
                    
                    
            if max_distance>self.min_distance:
                    camera_movement[frame_num]=[movement_x,movement_y]
                    previous_features=cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    
            previous_frame_gray=frame_gray.copy()
            
        if stub_path is not None:
            # Save tracks to stub
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            
        return camera_movement
    
    
    def draw_camera_movement(self,frames,camera_movement_per_frame):
        output_frames=[]
        for frame_num,frame in enumerate(frames):
            frame=frame.copy()
            overlay=frame.copy()
            
            cv2.rectangle(overlay, (0,0), (700,100), (255,255,255), -1)
            alpha=0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            movement_x , movement_y= camera_movement_per_frame[frame_num]
            frame=cv2.putText(frame, f'Camera Movement X: {movement_x:.2f} Y: {movement_y:.2f}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
            output_frames.append(frame)
        return output_frames