from tracker import Tracker
from utils import read_video, save_video
from analysis import TeamAssigner, PlayerBallAssigner, \
    CameraMovementEstimator, ViewTransformer, SpeedDistanceEstimator
import cv2 
import numpy as np

'''
# return frames of a video based on the given path
def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)

    return frames
    

# save the input frames at the specified path
def save_video(video_frames, video_path):
    writer = cv2.VideoWriter_fourcc(*'XVID')

    # fetch frame dimesions by analysing input frame
    output = cv2.VideoWriter(video_path, writer, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))

    for frame in video_frames:
        output.write(frame)
    
    output.release()
'''

def main():
    video_frames = read_video('input-videos/input-video1')

    # initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_file=True, file_path='tracks/tracks.pkl')

    # get object positions
    tracker.add_position_to_tracks(tracks)

    # handle camera movement
    cam_movement_estimator = CameraMovementEstimator(video_frames[0])
    cam_movement_per_frame = cam_movement_estimator.get_camera_movement(video_frames,
                                                                        read_from_file=True, 
                                                                        file_path='stubs/camera_movement.pkl')

    cam_movement_estimator.adjust_positions(tracks, cam_movement_per_frame)

    # perspective transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position(tracks)

    # interpolate ball 
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # speed and dist estimator
    speed_dist_estimator = SpeedDistanceEstimator()
    speed_dist_estimator.draw_speed_distance(output_video_frames, tracks)

    # assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # get team
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bounding_box'],
                                                 player_id)
            
            # save team in dict
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # assigning ball
    player_assigner = PlayerBallAssigner()
    ball_possession_team = [] # assigning a team name to each frame - tells the team in control of ball

    for frame_num, player_track in enumerate(tracks['players']):
        ball_box = tracks['ball'][frame_num][1]['bounding_box']
        assigned_player = player_assigner.assign_ball(player_track, ball_box)

        if assigned_player != -1: # -1 means no player assigned
            # set the has_ball parameter of the player to True
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            ball_possession_team.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            ball_possession_team.append(ball_possession_team[-1])

    ball_possession_team = np.array(ball_possession_team)
    # annotate input video
    output_video_frames = tracker.annotate(video_frames, tracks, ball_possession_team)

    # draw camera movement
    output_video_frames = cam_movement_estimator.draw_camera_movement(output_video_frames, cam_movement_per_frame)

    # save video
    save_video(output_video_frames, 'outputVideos/output_video1.avi')