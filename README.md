# Description

Implementing a smart tracker that captures player movements in a gameplay and provides game statistics using YOLO and OpenCV

# Features

Features include -

1. precise player, referee and ball tracking throughout the game
2. classifying players into teams based on jersey colors
3. special trackers on players controlling the ball at a given time
4. ball possession stats on how much each team controlled the ball
5. camera movement to capture accurate real world player movement
6. perspective transformation for player speed and distance calculation

Custom YOLOv5 training using Roboflow :

1. enhance image detection and differentiate referee and players from other non-playing people in the frame
2. continous ball detection and tracking throughout the video
