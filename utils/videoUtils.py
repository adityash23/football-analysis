import cv2

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