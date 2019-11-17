import numpy as np


def upper_left(v):
    """upper_left
    finds the lower right position based on an array cointaing in
    each row four values:
        [cx,cy,h,w]

    Args:
        v (np.array): each row [cx, cy, h, w] of a point

    Returns:
        np.array: array with x values of upper left
        np.array: array with y values of upper left
    """
    return v[:, 0] - v[:, 2] / 2, v[:, 1] - v[:, 3] / 2


def lower_right(v: np.array) -> (np.array, np.array):
    """lower_right
    finds the lower right position based on an array cointaing in
    each row four values:
        [cx,cy,h,w]

    Args:
        v (np.array): each row [cx, cy, h, w]

    Returns:
        np.array: array with x values of lower right
        np.array: array with y values of lower right
    """
    return v[:, 0] + v[:, 2] / 2, v[:, 1] + v[:, 3] / 2


def iou(a: np.array, b: np.array, epsilon: float = 1e-7) -> float:
    """ Given two arrays `a` and `b` where each row contains
        a bounding box defined as a list of four numbers:
            [cx,cy,h,w]
        where:
            cx represents center of bounding box in axis x
            cy represents center of bounding box in axis y
            h represents the height of bounding box
            w represents the width of bounding box
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a (np.array): each row containing [cx, cy, h, w]
        b (np.array): each row containing [cx, cy, h, w]

    Returns:
        float: iou calculated
    """

    # expand a and b array dimension
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=0)

    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=0)
        a = np.tile(a, (b.shape[0], 1))

    # finding the core corners
    a_ul = upper_left(a)
    a_lr = lower_right(a)

    b_ul = upper_left(b)
    b_lr = lower_right(b)

    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a_ul[0], b_ul[0]]).max(axis=0)
    y1 = np.array([a_ul[1], b_ul[1]]).max(axis=0)

    x2 = np.array([a_lr[0], b_lr[0]]).min(axis=0)
    y2 = np.array([a_lr[1], b_lr[1]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou
