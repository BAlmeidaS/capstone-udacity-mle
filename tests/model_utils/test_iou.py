from project.model_utils.iou import iou
import numpy as np
from numpy import testing


def test_iou_with_1_param():
    result = iou(np.array([2.05, 2.05, .2, .2]),
                 np.array([2, 2, .2, .2]))

    testing.assert_array_almost_equal(result,
                                      np.array([0.39130367]),
                                      decimal=7)


def test_iou_with_2_params():
    result = iou(np.array([[2.05, 2.05, .2, .2], [1, 1, 3, 3]]),
                 np.array([[2, 2, .2, .2], [1, 1, 3, 3]]))

    testing.assert_array_almost_equal(result,
                                      np.array([0.39130367, 1]),
                                      decimal=7)


def test_iou_without_intersection():
    result = iou(np.expand_dims(np.array([2.05, 2.05, .2, .2]), axis=0),
                 np.expand_dims(np.array([3, 3, .2, .2]), axis=0))

    testing.assert_array_almost_equal(result,
                                      np.array([0]),
                                      decimal=7)


def test_iou_with_unbalanced_params():
    result = iou(np.array([2.05, 2.05, .2, .2]),
                 np.array([[2, 2, .2, .2], [2, 2, .2, .2]]))

    testing.assert_array_almost_equal(result,
                                      np.array([0.39130367, 0.39130367]),
                                      decimal=7)
