from utils import read_video, save_video
from trackers import Tracker
import cv2
def main():
    #read video
    video_frames= read_video('input_videos/background-10.mp4')
    tracker=Tracker("models/best_yolo11l_100_640.pt")
    
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stub/yolo11l_stub.pkl')
    
    """
    for trackid,player in tracks["player"][0].items():
        bbox=player['bbox']
        frame=video_frames[0]
        
        #crop bbox from frame
        cropped_frame= frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        #save cropped frame
        cv2.imwrite("output_videos/cropped_player_0.jpg",cropped_frame)
        
        break
    """
        
    #draw output video
    output_frames=tracker.draw_annotations(video_frames,tracks)
    
    
    # save video
    save_video(output_frames, 'output_videos/yolo11l.avi')
    
    
if __name__=='__main__':
    main()