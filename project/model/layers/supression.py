import tensorflow as tf

from keras import layers
from project.model.loss import BBOX_REF

anchors = BBOX_REF.references.values


class CenterToCornerEncoder():
    def transform(bboxes: tf.Tensor) -> tf.Tensor:
        """transform
        based on a matrix with each row defined as [cx,cy,w,h] this encoder
        transforms to a new matrix with [y0,x0,y1,x1]

        Args:
            v (np.array): each row [cx, cy, w, h] of a point

        Returns:
            np.array: each row[y0, x0, y1, x1]
        """

        y0 = bboxes[..., -3] - tf.math.divide(bboxes[..., -1], 2)
        x0 = bboxes[..., -4] - tf.math.divide(bboxes[..., -2], 2)
        y1 = bboxes[..., -3] + tf.math.divide(bboxes[..., -1], 2)
        x1 = bboxes[..., -4] + tf.math.divide(bboxes[..., -2], 2)

        return tf.concat([tf.expand_dims(y0, axis=-1),
                          tf.expand_dims(x0, axis=-1),
                          tf.expand_dims(y1, axis=-1),
                          tf.expand_dims(x1, axis=-1)],
                         axis=-1)

    def inverse_transform(bboxes: tf.Tensor) -> tf.Tensor:
        """transform
        based on a matrix with each row defined as [y0,x0,y1,x1] this encoder
        transforms to a new matrix with [cx,cy,w,h]

        Args:
            v (np.array): each row [y0, x0, y1, x1] of a point

        Returns:
            np.array: each row[cx, cy, w, h]
        """
        h = bboxes[..., -2] - bboxes[..., -4]
        w = bboxes[..., -1] - bboxes[..., -3]
        cy = bboxes[..., -4] + tf.math.divide(h, 2)
        cx = bboxes[..., -3] + tf.math.divide(w, 2)

        return tf.concat([tf.expand_dims(cx, axis=-1),
                          tf.expand_dims(cy, axis=-1),
                          tf.expand_dims(w, axis=-1),
                          tf.expand_dims(h, axis=-1)],
                         axis=-1)


class Supression(layers.Layer):
    def call(self, inputs):
        return CenterToCornerEncoder.transform(inputs)
