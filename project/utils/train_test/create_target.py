import os

import numpy as np
import pandas as pd

import h5py

import project.download_content as content

from tqdm import tqdm

modelpath = os.path.join(content.DATAPATH, "MODEL")


def create_target(df, prefix):
    print(f'There are {len(df)} bounding boxes matched in {prefix}...')

    # setting path to data
    datapath = os.path.join(modelpath, f"{prefix}_data_300_vgg.h5")

    # sorting all df
    df = df.sort_values('ImageID')

    # open file and maintain it opened
    f = h5py.File(datapath, 'w')

    try:
        # get the first image
        img = df.iloc[0]

        # setting initial states to iterate over all dataframe
        img_name = img[0]
        img_path = img[7]
        images = []
        target = [img[13:-1].tolist() + img[9:13].tolist()]

        # iterate over all data set
        for i, img in tqdm(enumerate(df.iloc[:, :].itertuples())):
            # in first iteration of each batch size create the group
            # save last image when the new one is new
            if img_name != img[1]:
                images.append([img_name.encode("ascii", "ignore"),
                               img_path.encode("ascii", "ignore")])

                # create a dataset with the position and classification
                f.create_dataset(name=img_name,
                                 data=target,
                                 dtype=np.float32,
                                 compression='gzip',
                                 compression_opts=4)

                # clean all states
                target = []
                img_name = img[1]
                img_path = img[8]

            target.append(list(img[14:-1] + img[10:14]))

        f.create_dataset(name=img_name,
                         data=target[0],
                         dtype=np.float32,
                         compression='gzip',
                         compression_opts=4)

        f.create_dataset(name='images',
                         shape=(len(images), 2),
                         data=images,
                         dtype=h5py.special_dtype(vlen=str),
                         compression='gzip',
                         compression_opts=4)

    finally:
        f.close()


def main():
    filepath_val = os.path.join(modelpath, "cross_val_data_preprocessed.h5")
    create_target(pd.read_hdf(filepath_val, 'X', mode='r'), 'cross_val')

    filepath_test = os.path.join(modelpath, "test_data_preprocessed.h5")
    create_target(pd.read_hdf(filepath_test, 'X', mode='r'), 'test')


if __name__ == '__main__':
    main()
