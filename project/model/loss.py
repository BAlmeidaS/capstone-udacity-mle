import tensorflow as tf

from project.model.iou import iou
from project.model.smooth_l1 import smooth_l1


class SSDloss():
    def loss(self, y_true, y_pred, *args):
        loc = self.loc_loss(y_true, y_pred)

        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        loc_loss = tf.reduce_sum(loc * positives, axis=-1)

        return tf.convert_to_tensor(loc_loss, dtype=tf.float32)

    def x_k(self, ground_truth, default_box):
        return iou > 0.5

    def loc_loss(self, y_true, y_pred):
        z = y_true[:, :, -4:] - y_pred[:, :, -4:]
        return smooth_l1(z)
