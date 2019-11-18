import numpy as np
import tensorflow as tf


def smooth_l1(x: np.array) -> float:
    absolute_loss = tf.abs(x)
    square_loss = 0.5 * x ** 2

    l1_loss = tf.where(tf.less(absolute_loss, 1.0),
                       square_loss, absolute_loss - 0.5)

    return tf.reduce_sum(l1_loss)
