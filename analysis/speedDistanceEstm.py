import sys
sys.path.append('../')
from utils import distance, get_foot_position
import cv2

class SpeedDistanceEstimator():
    '''
    - to compute distance moved and speed of the user
    - computer every 5 frames
    '''
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance(self, tracks):
        total_distance = {}

        for obj, obj_tracks in tracks.items():
            if obj == 'ball' or obj == 'referee':
                continue # add speed and dist. info only for players
                
            number_of_frames = len(obj_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                # if total frames not a multiple of frame window, prevent exceeding total frames
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in obj_tracks[frame_num].items():
                    # if player not present in consecutive frames (spaced by 5) then don't compute stats
                    if track_id not in obj_tracks[last_frame]:
                        continue

                    start_position = obj_tracks[frame_num][track_id]['position_transformed']
                    end_position = obj_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = distance(start_position, end_position)

                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    speed_mps = distance_covered/time_elapsed # meters per second

                    speed_kmph = speed_mps * 3.6 # kilometers per hour

                    if obj not in total_distance:
                        total_distance[obj] = {}
                    
                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0

                    total_distance[obj][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue

                        tracks[obj][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[obj][frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]
    
    def draw_speed_distance(self, frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == 'ball' or obj == 'referee':
                    continue # add speed and dist. info only for players

                for _, track_info in obj_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        dist = track_info.get('distance', None)

                        if speed is None or dist is None:
                            continue

                        bounding_box = track_info['bounding_box']
                        position = get_foot_position(bounding_box)
                        position = list(position)

                        position[1] += 40 # give buffer in vertical direction to not overlap with circle or ID

                        position = tuple(map(int, position))

                        cv2.putText(frame, f'{speed:.2f} kmph', position, cv2.FONT_HERSHEY_COMPLEX, 
                                    0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f'{dist:.2f} m', (position[0], position[1] + 20), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
                        
            output_frames.append(frame)

        return output_frames