# imports
import os

import numpy as np
import pandas as pd

import project.download_content as content

from project.utils import data

from tqdm import tqdm

import pickle

from sklearn.preprocessing import OneHotEncoder


tqdm.pandas()
modelpath = os.path.join(content.DATAPATH, "MODEL")


def train_and_save_model(df):
    # train one hot encoding model
    ohc = OneHotEncoder(sparse=False, dtype=np.bool)
    ohc.fit(df[['LabelSemantic']])

    # save pickle
    with open(os.path.join(modelpath, 'ohc.pkl'), 'wb') as f:
        pickle.dump(ohc, f)

    return ohc


def main():
    all_data = data.all_train()

    ohc = train_and_save_model(all_data)

    labels = ohc.transform(all_data[['LabelSemantic']])

    all_data = all_data.join(pd.DataFrame(labels,
                                          columns=[c[3:] for c in ohc.get_feature_names()]))

    # isolate useful columns
    cols = ['boat', 'land_vehicle', 'skyscraper']
    part_data = (all_data.query(f"LabelSemantic in {cols}")[
        ['ImageID',
         'LabelName',
         'IsOccluded',
         'IsTruncated',
         'IsGroupOf',
         'IsDepiction',
         'IsInside',
         'Path',
         'LabelSemantic',
         'cx',
         'cy',
         'w',
         'h',
         'boat',
         'land_vehicle',
         'skyscraper']
    ])

    # define standard boxes (based on ssd architecture)
    standard_bboxes = data.StandardBoudingBoxes(feature_map_sizes=[38, 19, 10, 5, 3, 1],
                                                ratios_per_layer=[[1, 1/2, 2],
                                                                  [1, 1/2, 1/3, 2, 3],
                                                                  [1, 1/2, 1/3, 2, 3],
                                                                  [1, 1/2, 1/3, 2, 3],
                                                                  [1, 1/2, 2],
                                                                  [1, 1/2, 2]])

    # setting bbox ref which match with standard bboxes designed
    part_data['bbox_ref'] = part_data.progress_apply(standard_bboxes.match, axis=1)

    part_data.to_csv(modelpath + "/train_preprocessed.csv", index=False)


if __name__ == '__main__':
    main()
