from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os

# ---------------------------------------------------------------------------
# Standard pitch keypoint layout (metres, origin = top-left corner)
# Edit KEYPOINT_WORLD_COORDS to match your model's keypoint index order.
# Run a test prediction and visualise which index lands on which landmark.
# ---------------------------------------------------------------------------
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
PENALTY_BOX_WIDTH = 40.32
PENALTY_BOX_LENGTH = 16.5
GOAL_BOX_WIDTH = 18.32
GOAL_BOX_LENGTH = 5.5
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DIST = 11.0

KEYPOINT_WORLD_COORDS = {
    # Left Goal Line
    0: (0.0, 0.0),  # Top-left corner
    1: (0.0, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    2: (0.0, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),
    3: (0.0, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
    4: (0.0, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
    5: (0.0, PITCH_WIDTH),  # Bottom-left corner
    
    # Left Goal Box
    6: (GOAL_BOX_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),
    7: (GOAL_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
    
    # Left Penalty Spot
    8: (PENALTY_SPOT_DIST, PITCH_WIDTH / 2),
    
    # Left Penalty Box
    9: (PENALTY_BOX_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    10: (PENALTY_BOX_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),
    11: (PENALTY_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
    12: (PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
    
    # Halfway Line
    13: (PITCH_LENGTH / 2, 0.0),  # Top mid
    14: (PITCH_LENGTH / 2, PITCH_WIDTH / 2 - CENTRE_CIRCLE_RADIUS),
    15: (PITCH_LENGTH / 2, PITCH_WIDTH / 2 + CENTRE_CIRCLE_RADIUS),
    16: (PITCH_LENGTH / 2, PITCH_WIDTH),  # Bottom mid
    
    # Right Penalty Box
    17: (PITCH_LENGTH - PENALTY_BOX_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    18: (PITCH_LENGTH - PENALTY_BOX_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),
    19: (PITCH_LENGTH - PENALTY_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
    20: (PITCH_LENGTH - PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
    
    # Right Penalty Spot
    21: (PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_WIDTH / 2),
    
    # Right Goal Box
    22: (PITCH_LENGTH - GOAL_BOX_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),
    23: (PITCH_LENGTH - GOAL_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
    
    # Right Goal Line
    24: (PITCH_LENGTH, 0.0),  # Top-right corner
    25: (PITCH_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    26: (PITCH_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),
    27: (PITCH_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
    28: (PITCH_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
    29: (PITCH_LENGTH, PITCH_WIDTH),  # Bottom-right corner
    
    # Center Circle Edges
    30: (PITCH_LENGTH / 2 - CENTRE_CIRCLE_RADIUS, PITCH_WIDTH / 2),
    31: (PITCH_LENGTH / 2 + CENTRE_CIRCLE_RADIUS, PITCH_WIDTH / 2),
}


class ViewTransformer:
    """
    Wraps a YOLO pose model trained on pitch keypoints and handles:
      - Running the model over video frames
      - Computing per-frame homography matrices (pixel → world metres)
      - Transforming player/ball positions into world coordinates
      - Drawing a bird's-eye minimap overlay on output frames

    Usage
    -----
    vt = ViewTransformer("yolo26m_pose_weights/best.pt")
    homographies = vt.get_homographies(video_frames)
    vt.add_transformed_positions(tracks, homographies)

    # Inside your render loop:
    frame = vt.draw_minimap(frame, tracks, frame_num,
                             homographies[frame_num],
                             team_colours=team_assigner.team_colours)
    """

    def __init__(
        self,
        model_path: str,
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float  = PITCH_WIDTH,
    ):
        self.model        = YOLO(model_path)
        self.pitch_length = pitch_length
        self.pitch_width  = pitch_width

    # ------------------------------------------------------------------
    # 1. Run model → homographies  (call once, like Tracker)
    # ------------------------------------------------------------------

    def get_homographies(
    self,
    video_frames: list,
    predict_conf: float = 0.3,
    conf_threshold: float = 0.5,
    read_from_stub: bool = False,   # ← new
    stub_path: str = None,          # ← new
    ) -> list:
        # ── Load from stub if available ────────────────────────────────
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"      └─ Loading homographies from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # ── Run model ──────────────────────────────────────────────────
        batch_size = 20
        all_results = []
        for i in range(0, len(video_frames), batch_size):
            batch = self.model.predict(video_frames[i:i + batch_size],
                                    conf=predict_conf)
            all_results.extend(batch)

        homographies = []
        for result in all_results:
            H = self._homography_from_result(result, conf_threshold)
            homographies.append(H)

        valid = sum(H is not None for H in homographies)
        print(f"      └─ ViewTransformer: {valid}/{len(video_frames)} frames "
            f"have a valid homography")

        # ── Save to stub ───────────────────────────────────────────────
        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(homographies, f)
            print(f"      └─ Saved homographies to stub: {stub_path}")

        return homographies

    # ------------------------------------------------------------------
    # 2. Add world-space positions to every track entry
    # ------------------------------------------------------------------

    def add_transformed_positions(
        self,
        tracks: dict,
        homographies: list,
    ) -> None:
        """
        Adds 'position_transformed' (world metres) to every track entry
        for players, referees and ball. Uses 'position_adjusted' if
        available, otherwise falls back to 'position'.

        Modifies `tracks` in-place.
        """
        for obj_type in ('player', 'referee', 'ball'):
            if obj_type not in tracks:
                continue
            for frame_num, frame_data in enumerate(tracks[obj_type]):
                H = homographies[frame_num] if frame_num < len(homographies) else None
                for track_data in frame_data.values():
                    pos = (track_data.get('position')) or (track_data.get('position_adjusted'))
                    if pos is None or H is None:
                        track_data['position_transformed'] = None
                    else:
                        track_data['position_transformed'] = self._transform_point(pos, H)

    # ------------------------------------------------------------------
    # 3. Bird's-eye minimap overlay
    # ------------------------------------------------------------------

    def draw_minimap(
        self,
        frame: np.ndarray,
        tracks: dict,
        frame_num: int,
        H,
        minimap_origin: tuple = (20, None),   # None → auto bottom-left
        minimap_size: tuple   = (210, 140),
        alpha: float          = 0.75,
        team_colours: dict    = None,
    ) -> np.ndarray:
        """
        Draw a semi-transparent bird's-eye minimap onto `frame`.

        Parameters
        ----------
        frame          : BGR image (H×W×3 uint8)
        tracks         : standard tracks dict
        frame_num      : current frame index
        H              : homography for this frame (None → skip)
        minimap_origin : (x, y) top-left of the minimap; y=None → 20px from bottom
        minimap_size   : (width, height) in pixels
        alpha          : minimap opacity
        team_colours   : {team_id: (B,G,R)}
        """
        if H is None:
            return frame

        mw, mh = minimap_size
        mx, my_raw = minimap_origin
        my = (frame.shape[0] - mh - 20) if my_raw is None else my_raw

        overlay = frame.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (34, 85, 34), cv2.FILLED)
        self._draw_pitch_lines(overlay, mx, my, mw, mh)
        self._draw_entities(overlay, tracks, frame_num, H,
                            mx, my, mw, mh, team_colours or {})
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    # ------------------------------------------------------------------
    # Internal: homography from a single YOLO result
    # ------------------------------------------------------------------

    def _homography_from_result(self, result, conf_threshold: float):
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Mirroring your working script's extraction logic:
        coords = result.keypoints.xy[0].cpu().numpy()      # (num_keypoints, 2)
        kp_confs = result.keypoints.conf[0].cpu().numpy()  # (num_keypoints,)

        src_pts, dst_pts = [], []
        
        for idx, (point, conf) in enumerate(zip(coords, kp_confs)):
            x, y = int(point[0]), int(point[1])
            
            # 1. Skip if YOLO padded this keypoint with [0,0] (from your working script!)
            if x == 0 and y == 0:
                continue
                
            # 2. Skip if confidence is too low
            if float(conf) < conf_threshold:
                continue
                
            # 3. Skip if we haven't defined the real-world location for this index yet
            if idx not in KEYPOINT_WORLD_COORDS:
                continue
                
            src_pts.append([x, y])
            dst_pts.append(KEYPOINT_WORLD_COORDS[idx])

        # We need at least 4 valid points to calculate perspective shift
        if len(src_pts) < 4:
            return None

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H

    # ------------------------------------------------------------------
    # Internal: transform a single pixel point → world metres
    # ------------------------------------------------------------------

    def _transform_point(self, pixel_xy, H) -> tuple | None:
        if H is None:
            return None
        pt    = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)
        return (float(world[0, 0, 0]), float(world[0, 0, 1]))

    # ------------------------------------------------------------------
    # Internal: minimap helpers
    # ------------------------------------------------------------------

    def _world_to_minimap(self, world_xy, mx, my, mw, mh) -> tuple:
        px = int(mx + (world_xy[0] / self.pitch_length) * mw)
        py = int(my + (world_xy[1] / self.pitch_width)  * mh)
        return (px, py)

    def _draw_pitch_lines(self, img, mx, my, mw, mh) -> None:
        c  = (100, 180, 100)
        lw = 1

        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), c, lw)

        half_x = mx + mw // 2
        cv2.line(img, (half_x, my), (half_x, my + mh), c, lw)

        cx, cy = half_x, my + mh // 2
        cv2.circle(img, (cx, cy), int(min(mw, mh) * 0.13), c, lw)
        cv2.circle(img, (cx, cy), 2, c, cv2.FILLED)

        pb_d = int((16.5 / self.pitch_length) * mw)
        pb_h = int((40.32 / self.pitch_width) * mh)
        pb_t = cy - pb_h // 2
        cv2.rectangle(img, (mx,            pb_t), (mx + pb_d,      pb_t + pb_h), c, lw)
        cv2.rectangle(img, (mx + mw - pb_d, pb_t), (mx + mw,       pb_t + pb_h), c, lw)

    def _draw_entities(self, img, tracks, frame_num, H, mx, my, mw, mh, team_colours) -> None:

        def _plot(pos_pixel, colour, radius=4, outline=(0, 0, 0)):
            world = self._transform_point(pos_pixel, H)
            if world is None:
                return
            wx = max(0.0, min(self.pitch_length, world[0]))
            wy = max(0.0, min(self.pitch_width,  world[1]))
            mp = self._world_to_minimap((wx, wy), mx, my, mw, mh)
            cv2.circle(img, mp, radius,     outline, cv2.FILLED)
            cv2.circle(img, mp, radius - 1, colour,  cv2.FILLED)

        if 'player' in tracks and frame_num < len(tracks['player']):
            for data in tracks['player'][frame_num].values():
                pos =  data.get('position')
                if pos is None:
                    continue
                
                raw_colour = team_colours.get(data.get('team', 0), (220, 220, 220))
                
                # --- ADD THIS LINE ---
                # Force the raw color array into an integer tuple for OpenCV
                safe_colour = (int(raw_colour[0]), int(raw_colour[1]), int(raw_colour[2]))
                # ---------------------

                _plot(pos, safe_colour, radius=4) # Use safe_colour here

        if 'referee' in tracks and frame_num < len(tracks['referee']):
            for data in tracks['referee'][frame_num].values():
                pos = data.get('position_adjusted') or data.get('position')
                if pos is None:
                    continue
                _plot(pos, (0, 255, 255), radius=4)

        if 'ball' in tracks and frame_num < len(tracks['ball']):
            for data in tracks['ball'][frame_num].values():
                pos = data.get('position_adjusted') or data.get('position')
                if pos is None:
                    continue
                _plot(pos, (0, 220, 220), radius=5, outline=(0, 160, 160))