# imports
import os

import numpy as np
import pandas as pd

import project.download_content as content

from project.utils import data

from tqdm import tqdm

import pickle
import ray

from sklearn.preprocessing import OneHotEncoder
import multiprocessing


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


def load_model():
    with open(os.path.join(modelpath, 'ohc.pkl'), 'rb') as f:
        ohc = pickle.load(f)

    return ohc


@ray.remote
def part_process(df, part):
    # define standard boxes (based on ssd architecture)
    standard_bboxes = data.StandardBoudingBoxes(feature_map_sizes=[38, 19, 10, 5, 3, 1],
                                                ratios_per_layer=[[1, 1/2, 2],
                                                                  [1, 1/2, 1/3, 2, 3],
                                                                  [1, 1/2, 1/3, 2, 3],
                                                                  [1, 1/2, 1/3, 2, 3],
                                                                  [1, 1/2, 2],
                                                                  [1, 1/2, 2]])

    # setting bbox ref which match with standard bboxes designed
    df['bbox_ref'] = df.apply(standard_bboxes.match, axis=1)

    df.to_csv(modelpath + f"/train_preprocessed_{part}_4mto6m.csv", index=False)


def main():
    all_data = data.all_train()[4000000:6000000]

    # ohc = train_and_save_model(all_data)
    ohc = load_model()

    labels = ohc.transform(all_data[['LabelSemantic']])

    # removing useless prefix
    all_data = all_data.join(pd.DataFrame(labels,
                                          columns=[c[3:] for c in ohc.get_feature_names()]))

    # isolate useful columns
    # cols = ['boat', 'land_vehicle', 'skyscraper']
    # part_data = (all_data.query(f"LabelSemantic in {cols}")[
    #     ['ImageID',
    #      'LabelName',
    #      'IsOccluded',
    #      'IsTruncated',
    #      'IsGroupOf',
    #      'IsDepiction',
    #      'IsInside',
    #      'Path',
    #      'LabelSemantic',
    #      'cx',
    #      'cy',
    #      'w',
    #      'h',
    #      'boat',
    #      'land_vehicle',
    #      'skyscraper']
    # ])
    all_cols = ['ImageID', 'LabelName', 'IsOccluded', 'IsTruncated', 'IsGroupOf',
                'IsDepiction', 'IsInside', 'Path', 'LabelSemantic', 'cx', 'cy',
                'w', 'h'] + all_data.columns[19:].tolist()
    part_data = all_data[all_cols]

    parallel = multiprocessing.cpu_count() - 1

    step = int(part_data.shape[0] / parallel) + 1

    ray.init()
    try:
        futures = [part_process.remote(part_data[i*step:(1+i)*step], i) for i in range(parallel)]
        ray.get(futures)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
