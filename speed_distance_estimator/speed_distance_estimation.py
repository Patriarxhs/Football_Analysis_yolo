import numpy as np

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 25   # 1 second at 25fps — long enough to smooth homography noise
        self.max_speed_kmph = 35.0  # hard cap — no human runs faster than this

    def add_speed_and_distance_to_tracks(self, tracks, fps):
        total_distance = {}

        for object_type in ['player']:
            if object_type not in tracks:
                continue

            for frame_num in range(0, len(tracks[object_type]), self.frame_window):
                last_frame = min(frame_num + self.frame_window, len(tracks[object_type]) - 1)

                # skip if window is too short (avoids division by near-zero)
                if last_frame == frame_num:
                    continue

                for track_id, track_info in tracks[object_type][frame_num].items():
                    if track_id not in tracks[object_type][last_frame]:
                        continue

                    start_pos = track_info.get('position_transformed')
                    end_pos = tracks[object_type][last_frame][track_id].get('position_transformed')

                    if start_pos is None or end_pos is None:
                        continue

                    dist = np.linalg.norm(np.array(end_pos) - np.array(start_pos))

                    if track_id not in total_distance:
                        total_distance[track_id] = 0
                    total_distance[track_id] += dist

                    time_elapsed = (last_frame - frame_num) / fps
                    speed_mps = dist / time_elapsed
                    speed_kmph = speed_mps * 3.6

                    # suppress jitter (standing still)
                    if speed_kmph < 2.0:
                        speed_kmph = 0.0

                    # hard cap — physically impossible speeds are homography noise
                    if speed_kmph > self.max_speed_kmph:
                        speed_kmph = 0.0   # show nothing rather than a wrong number

                    for frame in range(frame_num, last_frame + 1):
                        if track_id in tracks[object_type][frame]:
                            tracks[object_type][frame][track_id]['speed'] = speed_kmph
                            tracks[object_type][frame][track_id]['distance'] = total_distance[track_id]