import numpy as np


def nms_numpy(boxes, scores, iou_threshold):
    """Non-Maximum Suppression (NMS) implemented in Numpy

    Args: 
        boxes (numpy.ndarray): Array of shape (N, 4) containing the bounding boxes.
        scores (numpy.ndarray): Array of shape (N, ) containing the corresponding scores/confidences.
        iou_threshold (float): IoU threshold for suppressing overlapping boxes.
    
    Returns:
        numpy.ndarray: Array of indices of the selected boxes after NMS
    """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1)*(y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])

        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        intersection = width*height

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


if __name__ == "__main__":
    boxes = np.array([[0.1, 0.2, 0.5, 0.6], 
                       [0.05, 0.15, 0.45, 0.55]])
    scores = np.array([0.8, 0.95])
    iou_threshold = 0.6

    print(nms_numpy(boxes, scores, iou_threshold))