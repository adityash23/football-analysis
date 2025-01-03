import sys
from utils import get_center

class Player_Ball_Assigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball(self, players, ball_bounding_box):
        ball_position = get_center(ball_bounding_box)
        