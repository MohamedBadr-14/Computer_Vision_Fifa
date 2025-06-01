import numpy as np

def get_center_of_bbox(bbox):
    """Calculate the center of a bounding box."""
    x_center = (bbox[0] + bbox[2]) // 2
    y_center = (bbox[1] + bbox[3]) // 2
    return x_center, y_center

def get_bbox_width(bbox):
    """Calculate the width of a bounding box."""
    return bbox[2] - bbox[0]

def get_foot_position(bbox):
    """Get the foot position (bottom center) of a bounding box."""
    x_center = (bbox[0] + bbox[2]) // 2
    y_bottom = bbox[3]
    return x_center, y_bottom

def measure_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))
