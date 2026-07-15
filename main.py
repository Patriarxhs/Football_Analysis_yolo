from utils import read_video, save_video
from trackers import Tracker
from team_assginer import Team_Assigner
from player_ball_assginger import PlayerBallAssigner
import cv2
import numpy as np

from camera_movement import CameraMovementEstimator
from view_tranformer import ViewTransformer
from speed_distance_estimator import SpeedAndDistanceEstimator

def main():
    # ── Read Video ──────────────────────────────────────────────────────────────
    input_path = 'input_videos/background-10.mp4'
    print(f"\n{'='*60}")
    print(f"  FOOTBALL ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"\n[1/8] 📂 Reading video...")
    print(f"      └─ Input : {input_path}")
    video_frames = read_video(input_path)
    print(f"      └─ Output: {len(video_frames)} frames loaded")

    # ── Object Tracking ─────────────────────────────────────────────────────────
    model_path  = "yolo11x/weights/best.pt"
    stub_path   = 'stub/yolo11x_stub_bck.pkl'
    print(f"\n[2/8] 🔍 Running object tracker...")
    print(f"      └─ Model : {model_path}")
    print(f"      └─ Stub  : {stub_path}")
    tracker = Tracker(model_path)
    tracks  = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
    n_players   = sum(len(f) for f in tracks['player'])
    n_referees  = sum(len(f) for f in tracks.get('referee', []))
    n_ball      = sum(len(f) for f in tracks['ball'])
    print(f"      └─ Output: tracks over {len(video_frames)} frames")
    print(f"                 players={n_players} detections | referees={n_referees} | ball={n_ball}")

    # ── Ball Interpolation ──────────────────────────────────────────────────────
    print(f"\n[3/8] ⚽ Interpolating ball positions...")
    missing_before = sum(1 for f in tracks['ball'] if not f)
    tracks['ball'] = tracker.interpollate_ball_postions(tracks['ball'])
    missing_after  = sum(1 for f in tracks['ball'] if not f)
    print(f"      └─ Frames without ball before : {missing_before}")
    print(f"      └─ Frames without ball after  : {missing_after}")

    # ── Position to Track ───────────────────────────────────────────────────────
    print(f"\n[4/8] 📍 Adding foot-position to tracks...")
    tracker.add_position_to_track(tracks)
    print(f"      └─ Done — 'position' key added to every track entry")

    # ── Camera Movement Estimation ──────────────────────────────────────────────
    cam_stub = 'stub/camera_movement_stub_bck.pkl'
    print(f"\n[5/8] 🎥 Estimating camera movement...")
    print(f"      └─ Reference frame : frame #0")
    print(f"      └─ Stub            : {cam_stub}")
    camera_movement_estimator    = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame    = camera_movement_estimator.get_camera_movement(
        video_frames, read_stub=True, stub_path=cam_stub
    )
    camera_movement_estimator.adjust_position(tracks, camera_movement_per_frame)
    total_movement = sum(abs(dx) + abs(dy) for dx, dy in camera_movement_per_frame)
    print(f"      └─ Output: movement vector per frame (total pixel drift ≈ {total_movement:.1f} px)")
    
    print(f"\n[5b/8] 🗺️  Running ViewTransformer (pitch keypoints → homography)...")
    view_transformer = ViewTransformer('yolo8l_pose/weights/best.pt')
    homographies = view_transformer.get_homographies(
    video_frames,
    read_from_stub=True,
    stub_path='stub/homographies_stub_bck.pkl',   
    )
    view_transformer.add_transformed_positions(tracks, homographies)
    
    for tid, data in list(tracks['player'][0].items())[:3]:
        print(f"[debug] player {tid}: position_transformed = {data.get('position_transformed')}")

    # ── Team Assignment ─────────────────────────────────────────────────────────
    print(f"\n[6/8] 👕 Assigning team colours...")
    print(f"      └─ Fitting KMeans on frame #0 player crops")
    team_assigner = Team_Assigner()
    team_assigner.assign_team_colour(video_frames[0], tracks["player"][0])
    print(f"      └─ Team colours: {team_assigner.team_colours}")

    team_counts = {1: 0, 2: 0}
    for frame_num, player_track in enumerate(tracks["player"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['player'][frame_num][player_id]['team']        = team
            tracks['player'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
            team_counts[team] = team_counts.get(team, 0) + 1
    print(f"      └─ Output: team label + colour assigned to every player track")
    print(f"                 Team 1 detections={team_counts.get(1,0)} | Team 2 detections={team_counts.get(2,0)}")

    # ── Ball Possession ─────────────────────────────────────────────────────────
    print(f"\n[7/8] 🏃 Assigning ball possession per frame...")
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control    = []

    team_passes = {1: 0, 2: 0} # Default fallback

    for frame_num, player_track in enumerate(tracks["player"]):
        ball_track = tracks['ball'][frame_num][1]
        
        # 1. Find who has the ball in this frame
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_track)

        current_player = None
        current_team = None

        if assigned_player != -1:
            tracks['player'][frame_num][assigned_player]['has_ball'] = True
            current_team = tracks['player'][frame_num][assigned_player]['team']
            team_ball_control.append(current_team)
            current_player = assigned_player
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else 0)

        # 2. Update Passes using our new clean class function!
        team_passes = player_ball_assigner.update_passes(current_player, current_team)


    # Calculate percentages
    team_ball_control_np = np.array(team_ball_control)
    t1_pct = (team_ball_control_np == 1).mean() * 100
    t2_pct = (team_ball_control_np == 2).mean() * 100
    
    print(f"      └─ Ball control — Team 1: {t1_pct:.1f}% | Team 2: {t2_pct:.1f}%")
    print(f"      └─ Passes Completed — Team 1: {team_passes.get(1, 0)} | Team 2: {team_passes.get(2, 0)}")
    
    

    print(f"\n[7b/8] ⏱️  Calculating speed and distance...")
    video_fps = 25  # Update this if your video is 30fps or 60fps!
    speed_estimator = SpeedAndDistanceEstimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks, video_fps)
    print(f"      └─ Done — 'speed' and 'distance' added to tracks")
    
    # ── Draw & Save ─────────────────────────────────────────────────────────────
    output_path = 'output_videos/yolo11x.avi'
    


    # check a sample frame's positions
    sample_frame = tracks['player'][0]
    for tid, data in list(sample_frame.items())[:3]:
        print(f"[debug] player {tid}: position={data.get('position')}, "
            f"position_adjusted={data.get('position_adjusted')}, "
            f"team={data.get('team')}, team_colour={data.get('team_colour')}")
    
    print(f"\n[8/8] 🎬 Rendering & saving output video...")
    output_frames = tracker.draw_annotations(
    video_frames,
    tracks,
    team_ball_control,
    view_transformer=view_transformer,   
    homographies=homographies,
    team_colours=team_assigner.team_colours,
    team_passes=team_passes,
    )
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)
    print(f"      └─ Annotated frames : {len(output_frames)}")
    save_video(output_frames, output_path)
    print(f"      └─ Saved to         : {output_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()