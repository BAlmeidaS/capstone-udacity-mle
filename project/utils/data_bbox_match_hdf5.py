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

import time

tqdm.pandas()
modelpath = os.path.join(content.DATAPATH, "MODEL")

standard_bboxes = data.StandardBoudingBoxes(feature_map_sizes=[38, 19, 10, 5, 3, 1],
                                            ratios_per_layer=[[1, 1/2, 2],
                                                              [1, 1/2, 1/3, 2, 3],
                                                              [1, 1/2, 1/3, 2, 3],
                                                              [1, 1/2, 1/3, 2, 3],
                                                              [1, 1/2, 2],
                                                              [1, 1/2, 2]])



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
    # setting bbox ref which match with standard bboxes designed
    df['bbox_ref'] = df.apply(standard_bboxes.match, axis=1)

    filepath = os.path.join(content.DATAPATH, "MODEL", f"data_preprocessed_{part}.h5")

    df.to_hdf(filepath, key='X', mode='w')


def main():
    all_data = data.all_train()

    # ohc = train_and_save_model(all_data)
    ohc = load_model()

    # removing useless prefix
    all_data = all_data.join(pd.DataFrame(ohc.transform(all_data[['LabelSemantic']]),
                                          columns=[c[3:] for c in ohc.get_feature_names()]))

    all_cols = ['ImageID', 'LabelName', 'IsOccluded', 'IsTruncated', 'IsGroupOf',
                'IsDepiction', 'IsInside', 'Path', 'LabelSemantic', 'cx', 'cy',
                'w', 'h'] + all_data.columns[19:].tolist()
    all_data = all_data[all_cols]

    parallel = multiprocessing.cpu_count()

    step = int((all_data.shape[0] / parallel) + 1)

    ray.init()
    try:
        futures = [part_process.remote(all_data[i*step:(1+i)*step], i) for i in range(parallel)]
        ray.get(futures)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
