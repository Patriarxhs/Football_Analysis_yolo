import sys
import numpy as np

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_distance = 1.5 
        
        # --- NEW: Memory for Pass Counting ---
        self.team_passes = {1: 0, 2: 0}
        self.last_player_with_ball = None
        self.last_team_with_ball = None
        self.consecutive_frames = 0
        self.REQUIRED_FRAMES = 5  # Anti-jitter threshold
        
    def assign_ball_to_player(self, players, ball_track):
        ball_position = ball_track.get('position_transformed')
        if ball_position is None:
            return -1
        
        min_distance = sys.maxsize
        assigned_player_id = -1
        
        for player_id, player in players.items():
            player_position = player.get('position_transformed')
            if player_position is None:
                continue
                
            distance = np.linalg.norm(np.array(player_position) - np.array(ball_position))
            
            if distance < min_distance and distance < self.max_player_distance:
                min_distance = distance
                assigned_player_id = player_id
                
        return assigned_player_id

    # --- NEW: Function to track passes ---
    def update_passes(self, current_player, current_team):
        if current_player is not None:
            if current_player == self.last_player_with_ball:
                # Same player is keeping the ball
                self.consecutive_frames += 1
            else:
                # The ball changed hands!
                # Did the previous player hold it long enough to avoid the "scramble bug"?
                if self.consecutive_frames >= self.REQUIRED_FRAMES and self.last_player_with_ball is not None:
                    # Are they on the same team?
                    if current_team == self.last_team_with_ball:
                        self.team_passes[current_team] += 1
                
                # Reset the tracker for the new player who just touched the ball
                self.last_player_with_ball = current_player
                self.last_team_with_ball = current_team
                self.consecutive_frames = 1
                
        return self.team_passes