import os
import numpy as np
import pandas as pd

import project.download_content as content
from project.model.iou import iou

METAPATH = os.path.join(content.DATAPATH, 'METADATA')


def all_train() -> pd.DataFrame:
    """Get all train data"""
    return pd.read_csv(METAPATH + '/enriched_train_bbox.csv')


class StandardBoudingBoxes:
    def __init__(self, feature_map_sizes, ratios_per_layer):
        if len(feature_map_sizes) != len(ratios_per_layer):
            raise RuntimeError('feature map size does not match with ratios')

        max_depth = len(feature_map_sizes)

        arr = []
        for i, (k, v) in enumerate(zip(feature_map_sizes, ratios_per_layer), 1):
            arr += self._bbox(k, v, i, max_depth)

        self.references = pd.DataFrame(arr, columns=['cx', 'cy', 'w', 'h'])

    def match(self, bbox: np.array, iou_threshold: float = 0.5) -> np.array:
        """return all bbox default that matches with some bbox passed

        Args:
            bbox (np.array): ['cx', 'cy', 'w', 'h']
            iou_threshold (float): iou threshold

        Returns:
            np.array
        """
        if len(bbox.shape) > 0 and bbox.shape[-1] != 4:
            raise ValueError('WRONG BBOX SHAPE')

        if len(bbox.shape) > 1:
            append = []
            for b in bbox:
                append = append + list(self.match(b, iou_threshold))
            return np.array(append)

        ious = iou(bbox, self.references.values)
        ious[ious < iou_threshold] = 0

        bboxs = np.where(ious > 0)[0]
        return bboxs

    def _bbox(self, feature_map_size, ratios, layer_depth, max_depth):
        arr = []

        step = 1/feature_map_size

        scale = self._func_scale(layer_depth, max_depth)
        next_scale = self._func_scale(layer_depth + 1, max_depth)

        for i in range(feature_map_size):
            for j in range(feature_map_size):
                cx = step * (1/2 + i)
                cy = step * (1/2 + j)

                for r in ratios:
                    w = step * scale * r ** (1/2)
                    h = step * scale / r ** (1/2)
                    arr.append([cx, cy, w, h])

                w = step * (scale*next_scale) ** (.5)
                h = step * (scale*next_scale) ** (.5)
                arr.append([cx, cy, w, h])
        return arr

    def _func_scale(self, layer_depth, max_depth):
        s_min = .2
        s_max = .9
        return s_min + ((s_max - s_min)/(max_depth - 1)) * (layer_depth - 1)
