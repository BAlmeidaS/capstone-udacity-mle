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
        def top():
            _, ids = tf.math.top_k(data[..., 1], k=k)
            return tf.gather(params=data, indices=ids, axis=0)
        return top

    def _pad_k(self, data, k):
        def pad():
            return tf.pad(data,
                          [[0, k - tf.shape(data)[0]], [0, 0]],
                          constant_values=0.0)
        return pad

    def call(self, inputs):
        centroids = CenterToCornerEncoder.transform(inputs)

        def filter_predictions(batch_item):
            mask = batch_item[..., :] > self.conf_threshold

            def filter_by_class(i):
                class_thresold = tf.boolean_mask(batch_item, mask[..., i])

                def suppresion():
                    identifier = tf.fill((tf.shape(class_thresold)[0], 1),
                                         tf.dtypes.cast(i, tf.float32))

                    preds = tf.concat([identifier,
                                       tf.gather(class_thresold,
                                                 [i, 600, 601, 602, 603],
                                                 axis=-1)],
                                      axis=-1)

                    suppression_ids = tf.image.non_max_suppression(
                        preds[..., -4:],
                        preds[..., 1],
                        max_output_size=self.nns_k,
                        iou_threshold=self.iou_threshold)

                    return tf.gather(preds, suppression_ids, axis=0)

                def empty():
                    return tf.constant(value=0.0, shape=(1, 6))

                relevants = tf.cond(tf.greater(tf.size(class_thresold), 0),
                                    suppresion,
                                    empty)

                return tf.cond(tf.greater_equal(tf.shape(relevants)[0],
                                                self.nns_k),
                               self._top_k(relevants, self.nns_k),
                               self._pad_k(relevants, self.nns_k))

            class_filter = tf.map_fn(fn=filter_by_class,
                                     elems=tf.range(1, 600),
                                     dtype=tf.float32,
                                     parallel_iterations=256,
                                     back_prop=False,
                                     swap_memory=False,
                                     infer_shape=True,
                                     name='loop_over_classes')

            results = tf.reshape(tensor=class_filter, shape=(-1, 6))

            top_results = tf.cond(tf.greater_equal(tf.shape(results)[0], self.k),
                                  self._top_k(results, self.k),
                                  self._pad_k(results, self.k))

            return top_results

        result = tf.map_fn(fn=filter_predictions,
                           elems=centroids,
                           dtype=None,
                           parallel_iterations=256,
                           back_prop=False,
                           swap_memory=False,
                           infer_shape=True,
                           name='loop_over_batch')
        return CenterToCornerEncoder.inverse_transform(result)
