from ultralytics import YOLO
import supervision  
import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../') # move 1 folder up in heirarchy
from utils import get_center, get_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = supervision.ByteTrack() # to prevent ID changes for objects 
        # ex - goal keeper changes to player in some frames

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        # do batch-wise analysis of frames
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf = 0.1)
            detections.append(detections_batch)
            break
        return detections

    def get_object_tracks(self, frames, read_from_file = False, file_path = None):
        # load tracks from existing pickle file
        if read_from_file and (file_path is not None) and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        ''' 
        dict of lists of dictionaries containing the bounding box of all the objects in the frame
        for all the frames in the video
        '''
        tracks = {
            'players' : [],
            'referees' : [],
            'ball' : []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # {0: person, 1: goal keeper ...}
            class_names_inv = {v : k for k,v in cls_names.items()} # {person : 0, goal keeper  : 1 ...}

            detection_supervision = supervision.Detections.from_ultralytics(detection)

            # convert goal keepr to player
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    # find the ID of player and change goal keeper accordingly - without hard coding IDs
                    detection_supervision.class_id[object_index] = class_names_inv['player'] 

            # tracking objects
            # assigning player number to each bounding box - to track players throughout the gameplay
            detection_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})

            for frame_detection in detection_tracks:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3] # index 3 found by looking at detection_tracks
                track_id = frame_detection[4]

                if class_id == class_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bounding_box' : bounding_box}

                if class_id == class_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bounding_box' : bounding_box}

            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bounding_box' : bounding_box} # only 1 ball so track_id hardcoded
            
            # save computed tracks into pickle file
            if file_path is not None:
                with open(file_path, 'wb') as f :
                    pickle.dump(tracks)

            return tracks
        

    def draw_ellipse(self, frame, bounding_box, color, track_id = None):
        y2 = int([bounding_box[3]]) # ellipse drawn beneath the player

        x_center, _ = get_center(bounding_box)
        width = get_width(bounding_box)

        cv2.ellipse(frame, 
        center  = (x_center, y2), 
        axes = (int(width), int(0.4 * width)), # major and minor axis of ellipse
        angle = 0.0,
        startAngle = -45,
        endAngle = 235, # not drawing the complete circle to leave space for players
        color = color,
        thickness = 2, 
        lineType = cv2.LINE_4)

        # making boxes for displaying player number
        rectangle_width = 40
        rectangle_height = 20

        # defining rectangle position in x,y,x,y format - top left and bottom right corners
        x1_rectangle = x_center - rectangle_width//2
        x2_rectangle = x_center + rectangle_width//2

        y1_rectangle = (y2 - rectangle_height//2) + 20
        y2_rectangle = (y2 + rectangle_height//2) + 20

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rectangle), int(y1_rectangle)),
                          (int(x2_rectangle), int(y2_rectangle)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rectangle + 13
            if track_id >= 100:
                x1_text -= 10

            cv2.putText(frame,
                        f'{track_id}',
                        (int(x1_text), int(y1_rectangle)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX(0,6, (0,0,0), 2))
            
        return frame


    def draw_triangle(self, frame, bounding_box, color):
        '''
        ball tracker looks like this - 
               ____
               \  /
                \/       
               ball  - bottom of triangle is at y-coordinate same as bounding box of ball    
        top left corner = move up in y, move left in x
        top right corner = move up in y, move right in x
        '''
        y = int(bounding_box[1]) 
        x, _ = get_center(bounding_box)

        triangle_points = np.array([[x,y],
                                    [x - 10, y - 20],
                                    [x + 10, y - 20]])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # draw triangle
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # draw boundary - non fill triangle with only the edges

        return frame

    def annotate(self, video_frames, tracks):
        output_frames = [] # video frames after creating the circles

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # copy to not modify the original video

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bounding_box'], (0, 0, 255), track_id)

            # draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bounding_box'], (0, 255, 255), track_id)
        
            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bounding_box'], (0, 255, 0))
                
            output_frames.append(frame)

        return output_frames