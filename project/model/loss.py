import tensorflow as tf

from project.model.smooth_l1 import smooth_l1


class SSDloss():
    @tf.function
    def loss(self, y_true, y_pred, *args):
        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=2)

        N = tf.reduce_sum(positives, axis=-1)[0]

        if tf.equal(N, 0):
            return tf.convert_to_tensor(0, dtype=tf.float32)

        loc = self.loc_loss(y_true, y_pred)
        conf_loss = self.conf_loss(y_true, y_pred)

        negatives = y_true[:, :, 0]
        positives = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)

        loc_loss = tf.reduce_sum(loc * tf.transpose(positives))

        loss = (conf_loss + 1 * loc_loss) / N

        return tf.convert_to_tensor(loss, dtype=tf.float32)

    def loc_loss(self, y_true, y_pred):
        z = y_true[:, :, -4:] - y_pred[:, :, -4:]
        return smooth_l1(z)

    def conf_loss(self, y_true, y_pred):
        ln = self.log(y_true, y_pred)
        return -tf.reduce_sum(ln * tf.transpose(y_true[:, :, :-4],
                                               perm=[0, 1, 2]))


    def log(self, y_true, y_pred):
        return tf.math.log(tf.math.maximum(y_pred[:, :, :-4],
                                           1e-8))
