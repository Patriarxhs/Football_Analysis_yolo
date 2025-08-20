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