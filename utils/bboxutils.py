def get_bbox_center(bbox):
    """
    Calculate the center of a bounding box.
    
    Args:
        bbox (list or tuple): Bounding box in the format [x1, y1, x2, y2].
        
    Returns:
        tuple: Center coordinates (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return int(cx), int(cy)


def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.
    
    Args:
        bbox (list or tuple): Bounding box in the format [x1, y1, x2, y2].
        
    Returns:
        int: Width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return int(x2 - x1)


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).
        
    Returns:
        float: Euclidean distance between the two points.
    """
    import math
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2) , int(y2)  # x center, y bottom 