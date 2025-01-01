def get_center(bounding_box):
    x1, y1, x2, y2 = bounding_box
    
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_width(boudning_box):
    return boudning_box[2] - boudning_box[0] # x2 - x1