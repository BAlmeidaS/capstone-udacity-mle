# imports
import os

import numpy as np

import project.download_content as content

from project.utils import data
from project.utils.data_bbox_match_hdf5 import load_ohc

from tqdm import tqdm

tqdm.pandas()
modelpath = os.path.join(content.DATAPATH, "MODEL")


def process_data(data, prefix):
    # loading one hot encode created previously
    ohc = load_ohc()

    # get one hot encode for each class
    encoded = ohc.transform(np.expand_dims(data['LabelSemantic'].values,
                                           axis=-1))
    # get the column id (position) of each row
    ref_ids = np.argmax(encoded, axis=1)

    # increase one to match with the scruct (NOCLASS, class1, class2, ...)
    class_ids = ref_ids + 1

    data['LabelID'] = class_ids
    data['Path'] = data.Path.apply(lambda x: "project/" + x[3:])

    filepath = os.path.join(content.DATAPATH, "MODEL", f"{prefix}_data_preprocessed.h5")

    data.to_hdf(filepath, key='X', mode='w')


def main():
    process_data(data.all_validation(), 'cross_val')
    process_data(data.all_test(), 'test')


if __name__ == '__main__':
    main()
