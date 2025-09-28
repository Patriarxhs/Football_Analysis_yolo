from utils import read_video, save_video
from trackers import Tracker
from team_assginer import Team_Assigner
from player_ball_assginger import PlayerBallAssigner
import cv2
import numpy as np
def main():
    #read video
    video_frames= read_video('input_videos/background-10.mp4')
    tracker=Tracker("yolo11x/weights/best.pt")
    
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stub/yolo11x_stub_bck.pkl')
    
    #interpolate ball positions
    tracks['ball']=tracker.interpollate_ball_postions(tracks['ball'])
    
    
    # for trackid,player in tracks["player"][0].items():
    #     if trackid==11:
    #         bbox=player['bbox']
    #         frame=video_frames[0]
            
    #         #crop bbox from frame
    #         cropped_frame= frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            
    #         #save cropped frame
    #         cv2.imwrite("output_videos/cropped_player_track_11.jpg",cropped_frame)
            
    #         break
    
    #assign team colours
    
    team_assigner=Team_Assigner()
    team_assigner.assign_team_colour(video_frames[0], tracks["player"][0])
    
    
    
    for frame_num,player_track in enumerate(tracks["player"]):
        for player_id,track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            
            tracks['player'][frame_num][player_id]['team'] = team
            tracks['player'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
            
            
    #assign ball to player
    player_ball_assigner=PlayerBallAssigner()
    team_ball_control=[]
    
    for frame_num,player_track in enumerate(tracks["player"]):
        ball_bbox=tracks['ball'][frame_num][1]['bbox'] #only one ball with id 1
        assigned_player=player_ball_assigner.assign_ball_to_player(player_track , ball_bbox)
        
        if assigned_player!=-1:
            tracks['player'][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
            
        
            
    team_ball_control=np.array(team_ball_control)
    #draw output video
    output_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)
    
    
    # save video
    save_video(output_frames, 'output_videos/yolo11x.avi')
    
    
if __name__=='__main__':
    main()