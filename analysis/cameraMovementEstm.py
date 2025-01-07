import pickle
import cv2
import numpy as np

import os
import sys
sys.path.append('../')
from utils import distance, xy_difference

class CameraMovementEstimator():
    '''
    - during camera movement, bounding boxes for players change even if the player 
    is at the same position relative to ground
    - poses a challenge if we calculate speed and distance moved for players
    - this class estimates camera movement and nullifies the extra motion in player movement
    - done by identifying pivots in the frame that are stationary 
    '''
    def __init__(self, frame):
        self.min_distance = 5 # min dist to consider camera moved

        self.lk_params = dict(
            windowSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
        mask_features = np.zeros_like(first_frame_gray)

        # put 1 at banner locations (top and bottom) - banners should stay fixed
        mask_features[:, 0 : 20] = 1
        mask_features[:, 900 : 1050] = 1

        self.features = dict(
            maxCorners = 100, 
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )


    def get_camera_movement(self, frames, read_from_file = False, file_path = None):
        # read file if already present
        if read_from_file and file_path is not None and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        # compare current frame with previous frame to compute movement
        prev_gray_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(prev_gray_frame, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # tracking features in the new frame w.r.t previous frame as per Lucas Kanade method
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray_frame, frame_gray, 
                                                          prev_features, None, **self.lk_params)
            
            max_distance = 0
            camera_movement_x, camera_movement_y, = 0, 0

            # IMP - to enumarate multiple lists, zip them together
            for i, (new, prev) in enumerate(zip(new_features, prev_features)):
                new_features_point = new.ravel()
                prev_features_point = prev.ravel()

                dist = distance(new_features_point, prev_features_point)

                if dist > max_distance:
                    max_distance = dist
                    camera_movement_x, camera_movement_y = xy_difference(prev_features_point, new_features_point)

            if max_distance > self.min_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                prev_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            prev_gray_frame = frame_gray.copy()

        # load the new analysis into file for future use
        if file_path is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # making box to display info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 1000), (255, 255, 255), cv2.FILLED)
            alpha = 0.6 # 60% transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f'Camera Movement X - {x_movement:.2f}', (10, 30), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f'Camera Movement Y - {y_movement:.2f}', (10, 60), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
            
            output_frames.append(frame)
        
        return output_frames