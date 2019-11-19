import tensorflow as tf

from project.model.iou import iou
from project.model.smooth_l1 import smooth_l1


class SSDloss():
    def __call__(self, y_true, y_pred):
        return self.loc_loss(y_true, y_pred)

    def x_k(self, ground_truth, default_box):
        return iou > 0.5

    def loc_loss(self, y_true, y_pred):
        z = y_true[:, :, -4:] - y_pred[:, :, -4:]

        smooth = smooth_l1(z)

        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        loc_loss = tf.reduce_sum(smooth * positives, axis=-1)

        return loc_loss
