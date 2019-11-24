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
    def __init__(self):
        self.set_references()

    def set_references(self, with_s2: list = [75, 39],
                       with_s1: list = [11, 7, 5, 3]) -> pd.DataFrame:
        """Get a df with all bbox defaults defined"""

        refs_s2 = self._bbox(with_s2, stride=2, ratios=[1/2])
        refs_s1 = self._bbox(with_s1, stride=1, ratios=[1/2, 1/3])

        self.references = (pd.concat([refs_s2, refs_s1])
                             .reset_index()[['cx', 'cy', 'w', 'h']])

    def match(self, bbox: pd.Series, iou_threshold: float = 0.5) -> np.array:
        """return all bbox default that matches with some bbox passed"""
        bbox = bbox[['cx', 'cy', 'w', 'h']]
        ious = iou(bbox.values, self.references.values)
        ious[ious < iou_threshold] = 0

        bboxs = np.where((ious == np.amax(ious)) & (ious > 0))[0]
        return bboxs

    def _bbox(self, sizes, stride=1, ratios=None):
        arr = []
        for n in sizes:
            e = 1 / n

            t = e*3

            for i in range(0, n-2, stride):
                for j in range(0, n-2, stride):
                    arr.append([e*(i+3/2), e*(j+3/2), t, t])

                    for r in ratios:
                        arr.append([e*(i+3/2), e*(j+3/2), t*r, t])
                    for r in ratios:
                        arr.append([e*(i+3/2), e*(j+3/2), t, t*r])
        return pd.DataFrame(arr, columns=['cx', 'cy', 'w', 'h'])
