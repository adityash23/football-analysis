import sys
from utils import get_center, distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball(self, players, ball_bounding_box):
        '''
        - computing distance of ball from the legs (bottom coordinates of bounding box) 
        of all players in the frame 
        - player with the least distance is assigned to the ball and has a tracker appearing on top
        '''
        ball_position = get_center(ball_bounding_box)

        # initializing values for dist and player id
        min_distance = 99999
        assigner_player = -1
        
        for player_id, player in players.items():
            player_box = player['bounding_box']

            distance_left = distance((player_box[0], player_box[-1]), ball_position)
            distance_right = distance((player_box[2], player_box[-1]), ball_position)

            dist = min(distance_left, distance_right)

            if dist < self.max_player_ball_distance:
                if dist < min_distance:
                    min_distance = dist
                    assigner_player = player_id

        return assigner_player