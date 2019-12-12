# imports
import os
os.sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd

import h5py

import project.download_content as content
from project.utils import data

from tqdm import tqdm

modelpath = os.path.join(content.DATAPATH, "MODEL")


def main():
    filepath = os.path.join(modelpath, "data_preprocessed.h5")
    all_train = pd.read_hdf(filepath, 'X', mode='r')
    all_train['bbox_count'] = all_train.bbox_ref.apply(lambda x: len(x))
    all_train = (pd.concat([all_train.query("bbox_count > 0")])
                   .sample(frac=1, random_state=17))

    imgs = all_train[['ImageID', 'Path']].drop_duplicates()
    # getting the one hot encodes columns
    dummy_classes = list(all_train.columns[13:-2])

    imgs.set_index('ImageID', inplace=True)

    print('creating list of each feature grouped by imageID...')

    for c in tqdm(['cx', 'cy', 'w', 'h', 'bbox_ref', 'LabelSemantic'] + dummy_classes):
        keys, values = all_train[['ImageID', c]].sort_values('ImageID').values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        imgs = imgs.join(pd.DataFrame({'ImageID': ukeys,
                                       c: [list(a) for a in arrays]})
                           .set_index('ImageID'))

    imgs.reset_index(inplace=True)

    # standard_bboxes = data.StandardBoudingBoxes(feature_map_sizes=[38, 19, 10, 5, 3, 1],
    #                                             ratios_per_layer=[[1, 1/2, 2],
    #                                                               [1, 1/2, 1/3, 2, 3],
    #                                                               [1, 1/2, 1/3, 2, 3],
    #                                                               [1, 1/2, 1/3, 2, 3],
    #                                                               [1, 1/2, 2],
    #                                                               [1, 1/2, 2]])

    print('creating target dataset...')

    # target = np.zeros((imgs.shape[0],
    #                   standard_bboxes.references.shape[0],
    #                   (1 + len(dummy_classes) + 4)),
    #                   dtype=np.float32)

    # target[:][:] = [1] + [0] * len(dummy_classes) + [0, 0, 0, 0]

    # for i, row in imgs.iterrows():
    #     for cx, cy, w, h, refs, *labels in zip(row.cx, row.cy, row.w, row.h, row.bbox_ref, *row[8:]):
    #         for id_ref in refs:
    #             target[i][int(id_ref)] = [0] + labels + [cx, cy, w, h]

    print('saving files...')

    filepath = os.path.join(content.DATAPATH, "MODEL", "all_data_300_vgg.h5")

    imgs.to_hdf(filepath, key='X', mode='w')

    # with h5py.File(filepath, 'a') as f:
    #     f.create_dataset('y', data=target, dtype=np.float16)


if __name__ == "__main__":
    main()
