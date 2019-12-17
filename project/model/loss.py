import tensorflow as tf

import numpy as np

from project.model.smooth_l1 import smooth_l1
from project.utils import data

BBOX_REF = data.StandardBoudingBoxes(feature_map_sizes=[38, 19, 10, 5, 3, 1],
                                     ratios_per_layer=[[1, 1/2, 2],
                                                       [1, 1/2, 1/3, 2, 3],
                                                       [1, 1/2, 1/3, 2, 3],
                                                       [1, 1/2, 1/3, 2, 3],
                                                       [1, 1/2, 2],
                                                       [1, 1/2, 2]])

STANDARD_BBOXES = np.expand_dims(BBOX_REF.references, axis=0)

MINCONST = 1e-15

verbose = 1


class SSDloss():
    @tf.function
    def loss(self, y_true, y_pred, *args):
        positives = tf.reduce_sum(y_true[:, :, 1:-4], axis=1)

        N = tf.reduce_sum(positives)

        if N == 0:
            return tf.convert_to_tensor(0, dtype=tf.float32)

        loc = self.loc_loss(y_true, y_pred)
        conf_loss_pos, amount_pos = self.conf_loss_pos(y_true, y_pred)
        conf_loss_neg = self.conf_loss_neg(y_true, y_pred, amount_pos)

        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        loc_loss = tf.reduce_sum(loc * positives)

        loss = (conf_loss_pos + conf_loss_neg + 1 * loc_loss) / N
        if verbose >= 1 and tf.random.uniform((1,))[0] > .99:
            tf.print()
            if verbose >= 2:
                tf.print('pred: ', y_pred[:, :, -4:])
                tf.print('loc_loss: ', loc_loss)
            tf.print('conf_loss_pos: ', conf_loss_pos)
            tf.print('conf_loss_neg: ', conf_loss_neg)
            tf.print('loc_loss: ', loc_loss)
            tf.print('bboxes num: ', N)
            tf.print('loss: ', loss)
            tf.print('\n')

        return tf.convert_to_tensor(loss, dtype=tf.float32)

    def g_hat(self, loc):
        g_hat_cx = 5 * tf.math.divide_no_nan(tf.math.subtract(loc[:, :, 0],
                                                          STANDARD_BBOXES[:, :, 0]),
                                         STANDARD_BBOXES[:, :, 2])

        g_hat_cy = 5 * tf.math.divide_no_nan(tf.math.subtract(loc[:, :, 1],
                                                          STANDARD_BBOXES[:, :, 1]),
                                         STANDARD_BBOXES[:, :, 3])

        g_hat_w = 10 * tf.math.log(
            tf.math.maximum(tf.math.divide_no_nan(loc[:, :, 2],
                                                  STANDARD_BBOXES[:, :, 2]),
                            MINCONST))

        g_hat_h = 10 * tf.math.log(
            tf.math.maximum(tf.math.divide_no_nan(loc[:, :, 3],
                                                  STANDARD_BBOXES[:, :, 3]),
                            MINCONST))

        return tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)

    def loc_loss(self, y_true, y_pred):
        z = y_pred[:, :, -4:] - self.g_hat(y_true[:, :, -4:])
        loc = tf.reduce_sum(smooth_l1(z), axis=-1)
        return tf.where(tf.math.is_nan(loc), tf.zeros_like(loc), loc)

    def conf_loss_neg(self, y_true, y_pred, amount_pos):
        ln = -1 * y_true[:, :, 0] * self.log(y_pred[:, :, 0])
        ln_flatten = tf.reshape(ln, (-1,))

        top_k, _ = tf.math.top_k(ln_flatten, k=tf.cast((amount_pos * 3), tf.int32))

        return tf.reduce_sum(top_k)

    def conf_loss_pos(self, y_true, y_pred):
        ln = -1 * y_true[:, :, 1:-4] * self.log(y_pred[:, :, 1:-4])
        return (tf.reduce_sum(ln),
                tf.math.count_nonzero(y_true[:, :, 1:-4]))

    def log(self, y_pred):
        return tf.math.log(tf.math.maximum(y_pred, MINCONST))
