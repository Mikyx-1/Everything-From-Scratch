import numpy as np
# Tested successfully

def calculate_iou(box1, box2):
    """Intersection Over Union implemented in Numpy
    
    Args: 
        box1 (numpy.ndarray): [x1, y1, x2, y2]
        box2 (numpy.ndarray): [x1, y1, x2, y2]
     
    Returns:
        numpy.array: Array of indices of the selected boxes after NMS.
    """    
    
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    x11 = box2[0] if box2[0] > box1[0] else box1[0]
    y11 = box2[1] if box2[1] > box1[1] else box1[1]

    x22 = box1[2] if box1[2] < box2[2] else box2[2]
    y22 = box1[3] if box1[3] < box2[3] else box2[3]

    intersection_area = (x22-x11)*(y22-y11)
    return (intersection_area)/(area1 + area2 - intersection_area)



if __name__ == "__main__":
    box1 = np.array([0.5, 0.5, 0.6, 0.6])
    box2 = np.array([0.5, 0.5, 0.7, 0.7])

    print(calculate_iou(box1, box2))