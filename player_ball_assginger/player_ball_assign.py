import sys
sys.path.append("../")
from utils import get_bbox_center, get_bbox_width,calculate_distance


class PlayerBallAssigner():
    def __init__(self):
        self.max_player_distance=70 #in pixels
        
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position=get_bbox_center(ball_bbox)
        
        min_distance=sys.maxsize
        assigned_player_id=-1
        
        for player_id,player in players.items():
            player_bbox=player["bbox"]
            distance_left_foot=calculate_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right_foot=calculate_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance=min(distance_left_foot,distance_right_foot)
            
            
            if distance<min_distance and distance<self.max_player_distance:
                min_distance=distance
                assigned_player_id=player_id
               
        
              
        return assigned_player_id