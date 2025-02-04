def get_center(bounding_box):
    x1, y1, x2, y2 = bounding_box
    
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_width(boudning_box):
    return boudning_box[2] - boudning_box[0] # x2 - x1

def distance(p1, p2):
    sum = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return sum ** 0.5

def xy_difference(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2)/2), y2