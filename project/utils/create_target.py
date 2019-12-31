# imports
import os

import numpy as np
import pandas as pd

import h5py

import project.download_content as content

from tqdm import tqdm

modelpath = os.path.join(content.DATAPATH, "MODEL")


def main():
    filepath = os.path.join(modelpath, "data_preprocessed.h5")
    all_train = pd.read_hdf(filepath, 'X', mode='r')

    print(f"There are {len(all_train)} bounding boxes matched...")

    # setting path to data
    datapath = os.path.join(modelpath, "images_bboxes.h5")

    # sorting all train df
    all_train = all_train.sort_values('ImageID')

    # open file and maintain it opened
    f = h5py.File(datapath, 'w')

    # size of the hfd5 group
    group_size = 300000

    try:
        # get the first image
        img = all_train.iloc[0]

        # setting initial states to iterate over all dataframe
        img_name = img[0]
        img_path = img[7]
        images = []
        target = [img[13:-1].tolist() + img[9:13].tolist()]

        # iterate over all data set
        for i, img in tqdm(enumerate(all_train.iloc[:, :].itertuples())):
            # in first iteration of each batch size create the group
            if i % group_size == 0:
                group = str(int(i/group_size))
                f.create_group(group)

            # save last image when the new one is new
            if img_name != img[1]:
                images.append([img_name.encode("ascii", "ignore"),
                              img_path.encode("ascii", "ignore")])

                # create a dataset with the position and classification
                f[group].create_dataset(name=img_name,
                                        data=target,
                                        dtype=np.float32,
                                        compression='gzip',
                                        compression_opts=4)

                # clean all states
                target = []
                img_name = img[1]
                img_path = img[8]

            # save the images dataset in last iteration of each batch
            if (i+1) % group_size == 0:
                f[group].create_dataset(name='images',
                                        shape=(len(images), 2),
                                        data=images,
                                        dtype=h5py.special_dtype(vlen=str),
                                        compression='gzip',
                                        compression_opts=4)
                images = []

            target.append(list(img[14:-1] + img[10:14]))

        f[group].create_dataset(name=img_name,
                                data=target[0],
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4)

        f[group].create_dataset(name='images',
                                shape=(len(images), 2),
                                data=images,
                                dtype=h5py.special_dtype(vlen=str),
                                compression='gzip',
                                compression_opts=4)

    finally:
        f.close()


if __name__ == '__main__':
    main()
