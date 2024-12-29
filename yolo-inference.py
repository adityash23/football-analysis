from ultralytics import YOLO

model = YOLO('yolov10m')

resultsVideo1 = model.predict('input-videos/input-video1.mp4', save = True)

#print(resultsVideo1[0])
'''
print('=========================')
for box in resultsVideo1[0].boxes:
    print(box)
'''