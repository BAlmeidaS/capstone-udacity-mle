import numpy as np
from project.model_utils.smooth import smooth_l1


def test_smooth_l1():
    result = smooth_l1(np.array([-.5]))
    assert result == 0.125


def test_smooth_l1_with_more_params():
    result = smooth_l1(np.array([1, -.5, -2]))
    assert result == 2.125
