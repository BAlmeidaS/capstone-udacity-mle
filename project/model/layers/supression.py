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

        return tf.concat([bboxes[..., :-4],
                          tf.expand_dims(y0, axis=-1),
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

        return tf.concat([bboxes[..., :-4],
                          tf.expand_dims(cx, axis=-1),
                          tf.expand_dims(cy, axis=-1),
                          tf.expand_dims(w, axis=-1),
                          tf.expand_dims(h, axis=-1)],
                         axis=-1)


class Suppression(layers.Layer):
    def __init__(self,
                 conf_threshold=0.01,
                 iou_threshold=0.45,
                 nns_k=300,
                 k=200,
                 **kwargs):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nns_k = nns_k
        self.k = k
        super().__init__(**kwargs)

    def _top_k(self, data, k):
        _, ids = tf.math.top_k(data[..., 1], k=k)
        return tf.gather(params=data, indices=ids, axis=0)

    def call(self, inputs):
        extended = tf.expand_dims(inputs, axis=-2)

        bboxes, scores, classes, _ = tf.image.combined_non_max_suppression(
            extended[..., -4:],
            inputs[:, :, 1:-4],
            self.nns_k,
            self.k,
            iou_threshold=self.iou_threshold,
            score_threshold=self.conf_threshold,
        )

        # adjusting class reference - class=0 is 'NoClass'
        classes = classes + 1

        return tf.concat([tf.expand_dims(classes, axis=-1),
                          tf.expand_dims(scores, axis=-1),
                          bboxes],
                         axis=-1)
