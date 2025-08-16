from utils import read_video, save_video
from trackers import Tracker
def main():
    #read video
    video_frames= read_video('input_videos/test.mp4')
    tracker=Tracker("models/best_yolo11l_100_640.pt")
    
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stub/track_stub.pkl')
    
    
    
    #draw output video
    output_frames=tracker.draw_annotations(video_frames,tracks)
    
    
    # save video
    save_video(output_frames, 'output_videos/output_video_try_test_video.avi')
    
    
if __name__=='__main__':
    main()