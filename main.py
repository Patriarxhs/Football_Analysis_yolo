from utils import read_video, save_video
from trackers import Tracker
from team_assginer import Team_Assigner
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
    #assign team colours
    team_assigner=Team_Assigner()
    team_assigner.assign_team_colour(video_frames[0], tracks["player"][0])
    
    for frame_num,player_track in enumerate(tracks["player"]):
        for player_id,track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            
            tracks['player'][frame_num][player_id]['team'] = team
            tracks['player'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
    
    #draw output video
    output_frames=tracker.draw_annotations(video_frames,tracks)
    
    
    # save video
    save_video(output_frames, 'output_videos/yolo11l.avi')
    
    
if __name__=='__main__':
    main()