import os
import pickle
import cv2
import numpy as np


class CameraMovementEstimator:
    def __init__(self,frame,):
        first_frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features=cv2.goodFeaturesToTrack(first_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    
    def get_camera_movement(self,frames,read_stub=False,stub_path=None):
        #read from stub
        if read_stub==True and stub_path is not None and os.path.exists(stub_path):
            # Load tracks from stub if it exists
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
        
        camera_movement=[[0,0]*len(frames)] #x,y movement for each frame
        
        previous_frame_gray=cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        previous_features=cv2.goodFeaturesToTrack(previous_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)