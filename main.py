from tracker import Tracker
from utils import read_video, save_video
from analysis import TeamAssigner
import cv2

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

    # interpolate ball 
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

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

    # annotate input video
    output_video_frames = tracker.annotate(video_frames, tracks)

    save_video(output_video_frames, 'outputVideos/output_video1.avi')