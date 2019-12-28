# imports
import os

import pandas as pd

import project.download_content as content

from project.utils import data
from project.model.loss import BBOX_REF as standard_bboxes

from tqdm import tqdm

import pickle
import ray

import multiprocessing

tqdm.pandas()
modelpath = os.path.join(content.DATAPATH, "MODEL")


def load_model():
    with open(os.path.join(modelpath, 'ohc.pkl'), 'rb') as f:
        ohc = pickle.load(f)

    return ohc


@ray.remote
def part_process(df, part, prefix):
    # define standard boxes (based on ssd architecture)
    # setting bbox ref which match with standard bboxes designed
    df['bbox_ref'] = df[['cx', 'cy', 'w', 'h']].apply(lambda x: standard_bboxes.match(x.values),
                                                      axis=1)

    filepath = os.path.join(content.DATAPATH, "MODEL", f"{prefix}_data_preprocessed_{part}.h5")

    df.to_hdf(filepath, key='X', mode='w')


def process_data(data, prefix):
    # loading one hot encode created previously
    ohc = load_model()

    # removing useless prefix
    data = data.join(pd.DataFrame(ohc.transform(data[['LabelSemantic']]),
                                  columns=[c[3:] for c in ohc.get_feature_names()]))

    all_cols = ['ImageID', 'LabelName', 'IsOccluded', 'IsTruncated', 'IsGroupOf',
                'IsDepiction', 'IsInside', 'Path', 'LabelSemantic', 'cx', 'cy',
                'w', 'h'] + data.columns[19:].tolist()
    data = data[all_cols]

    parallel = multiprocessing.cpu_count()

    step = int((data.shape[0] / parallel) + 1)

    ray.init()
    try:
        futures = [part_process.remote(data[i*step:(1+i)*step], i, prefix) for i in range(parallel)]
        ray.get(futures)
    finally:
        ray.shutdown()


def main():
    process_data(data.all_validation(), 'cross_val')
    process_data(data.all_test(), 'test')


if __name__ == '__main__':
    main()
