import numpy as np


def smooth_l1(x: np.array) -> float:
    absolute_loss = np.abs(x)
    square_loss = 0.5 * x ** 2

    l1_loss = np.where(np.less(absolute_loss, 1.0),
                       square_loss, absolute_loss - 0.5)

    return np.sum(l1_loss)
