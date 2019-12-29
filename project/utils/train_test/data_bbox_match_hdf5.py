# imports
import os

import pandas as pd

import project.download_content as content

from project.utils import data

from tqdm import tqdm

import pickle

tqdm.pandas()
modelpath = os.path.join(content.DATAPATH, "MODEL")


def load_model():
    with open(os.path.join(modelpath, 'ohc.pkl'), 'rb') as f:
        ohc = pickle.load(f)

    return ohc


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

    filepath = os.path.join(content.DATAPATH, "MODEL", f"{prefix}_data_preprocessed.h5")

    data.to_hdf(filepath, key='X', mode='w')


def main():
    process_data(data.all_validation(), 'cross_val')
    process_data(data.all_test(), 'test')


if __name__ == '__main__':
    main()
