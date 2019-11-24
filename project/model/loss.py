import tensorflow as tf

from project.model.smooth_l1 import smooth_l1


class SSDloss():
    @tf.function
    def loss(self, y_true, y_pred, *args):
        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        N = tf.reduce_sum(positives)

        if N == 0:
            tf.convert_to_tensor(1e-15, dtype=tf.float32)

        loc = self.loc_loss(y_true, y_pred)
        conf = self.conf_loss(y_true, y_pred)

        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        loc_loss = tf.reduce_sum(loc * positives, axis=-1)

        conf_loss_pos = tf.reduce_sum(conf * positives, axis=-1)
        conf_loss_neg = tf.reduce_sum(conf * negatives, axis=-1)
        conf_loss = conf_loss_pos + conf_loss_neg

        loss = (1/N) * (conf_loss + 1 * loc_loss)

        return loss
        # return tf.convert_to_tensor(loss, dtype=tf.float32)

    def loc_loss(self, y_true, y_pred):
        z = y_true[:, :, -4:] - y_pred[:, :, -4:]
        return smooth_l1(z)

    def conf_loss(self, y_true, y_pred):
        return -tf.reduce_sum(y_true[:, :, :-4]
                              * tf.math.log(y_pred[:, :, :-4]), axis=-1)
