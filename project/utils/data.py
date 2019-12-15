import os
import numpy as np
import pandas as pd

import project.download_content as content
from project.model.iou import iou

METAPATH = os.path.join(content.DATAPATH, 'METADATA')


def all_train() -> pd.DataFrame:
    """Get all train data"""
    return pd.read_csv(METAPATH + '/enriched_train_bbox.csv')


def boats() -> pd.DataFrame:
    """Get all train data with boats in bounding boxes"""
    return all_train().query("LabelSemantic == 'Boat'")


def group_images(df: pd.DataFrame) -> pd.DataFrame:
    """return a dataframe with all images grouped"""
    imgs = df[['ImageID', 'Path']].drop_duplicates()
    imgs.set_index('ImageID', inplace=True)

    for c in ['cx', 'cy', 'w', 'h', 'bbox_ref']:
        imgs = imgs.join(boats.groupby('ImageID')[c].apply(list))

    imgs.reset_index(inplace=True)
    return imgs


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

