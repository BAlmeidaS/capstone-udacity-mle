import tensorflow as tf

from project.model.smooth_l1 import smooth_l1


class SSDloss():
    @tf.function
    def loss(self, y_true, y_pred, *args):
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        N = tf.reduce_sum(positives, axis=-1)[0]

        if tf.equal(N, 0):
            return tf.convert_to_tensor(0, dtype=tf.float32)

        loc = self.loc_loss(y_true, y_pred)
        conf_loss_pos, amount_pos = self.conf_loss_pos(y_true, y_pred)
        conf_loss_neg = self.conf_loss_neg(y_true, y_pred, amount_pos)

        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        loc_loss = tf.reduce_sum(loc * positives)

        loss = (conf_loss_pos + conf_loss_neg + 1 * loc_loss) / N

        return tf.convert_to_tensor(loss, dtype=tf.float32)

    def loc_loss(self, y_true, y_pred):
        z = y_pred[:, :, -4:] - y_true[:, :, -4:]
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
        return tf.math.log(tf.math.maximum(y_pred, 1e-15))
