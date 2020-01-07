import tensorflow as tf

from keras import layers
from project.model.loss import BBOX_REF

anchors = BBOX_REF.references.values


class Anchorage(layers.Layer):
    def call(self, inputs):
        cx = tf.expand_dims(tf.math.multiply(inputs[..., -4],
                                             anchors[..., -2])
                            + anchors[..., -4],
                            axis=-1)
        cy = tf.expand_dims(tf.math.multiply(inputs[..., -3],
                                             anchors[..., -1])
                            + anchors[..., -3],
                            axis=-1)
        w = tf.expand_dims(tf.math.multiply(tf.math.exp(inputs[..., -2]),
                                            anchors[..., -2]),
                           axis=-1)
        h = tf.expand_dims(tf.math.multiply(tf.math.exp(inputs[..., -1]),
                                            anchors[..., -1]),
                           axis=-1)

        return tf.concat(values=[inputs[..., :-4], cx, cy, w, h], axis=-1)
