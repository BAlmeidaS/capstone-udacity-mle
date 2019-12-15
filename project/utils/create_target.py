# imports
import os

import numpy as np
import pandas as pd

import h5py

import project.download_content as content
from project.utils.data_bbox_match_hdf5 import standard_bboxes

from tqdm import tqdm

modelpath = os.path.join(content.DATAPATH, "MODEL")


def main():
    filepath = os.path.join(modelpath, "data_preprocessed.h5")
    all_train = pd.read_hdf(filepath, 'X', mode='r')

    # adding a column with amount of bbox matched
    all_train['bbox_count'] = all_train.bbox_ref.apply(lambda x: len(x))

    # using only bboxes that have a match with some default bbox
    all_train = all_train[all_train.bbox_count > 0]

    print(f"There are {len(all_train)} bounding boxes matched...")

    datapath = os.path.join(modelpath, "data_300_vgg.h5")

    # getting all image names
    imgs = all_train['ImageID'].unique()

    print(f"There are {len(imgs)} images...")
    # encoding each one with ascii
    ascii_imgs = [n.encode("ascii", "ignore") for n in imgs]

    # saving in dataset called 'images'
    with h5py.File(datapath, 'w') as f:
        f.create_dataset('images', (len(ascii_imgs),),
                         f'S{len(ascii_imgs[0])}', ascii_imgs)

    # getting all columns class
    dummy_classes = list(all_train.columns[13:-2])

    # sorting all train df
    all_train = all_train.sort_values('ImageID')

    f = h5py.File(datapath, 'a')

    dtype = np.float16

    try:
        img_name = all_train.iloc[0].ImageID

        target = np.zeros((standard_bboxes.references.shape[0],
                           (1 + len(dummy_classes) + 4)),
                          dtype=dtype)
        # set all bboxes default with 1 in no_class
        target[:][:] = [1] + [0] * len(dummy_classes) + [0, 0, 0, 0]

        for img in tqdm(all_train.iloc[:, :].itertuples()):
            if img_name != img[1]:
                f.create_dataset(img_name,
                                 shape=(standard_bboxes.references.shape[0],
                                        (1 + len(dummy_classes) + 4)),
                                 compression='gzip', compression_opts=1,
                                 data=target,
                                 dtype=dtype)

                target = np.zeros((standard_bboxes.references.shape[0],
                                   (1 + len(dummy_classes) + 4)),
                                  dtype=dtype)

                # set all bboxes default with 1 in no_class
                target[:][:] = [1] + [0] * len(dummy_classes) + [0, 0, 0, 0]

                img_name = img[1]

            # iterate over each bbox ref
            for bbox_id in img[-2]:
                # fill the target with the following logic:
                # no_class + one_hot_enc_of_each_class + cx + cy + w + h
                target[bbox_id] = np.array([0,
                                            *img[14:-2],
                                            *img[10:14]])

    finally:
        f.close()


def adding_path_in_image():
    filepath = os.path.join(modelpath, "data_preprocessed.h5")
    all_train = pd.read_hdf(filepath, 'X', mode='r')

    datapath = os.path.join(modelpath, "data_300_vgg.h5")

    # reading old dataset
    with h5py.File(datapath, 'r') as f:
        images = [i.decode() for i in f['images'][:]]

    # create a DF pandas with image path
    img_names = pd.DataFrame(images, columns=['ImageID'])
    new_refs = img_names.merge(all_train[['ImageID', 'Path']].drop_duplicates(),
                               on='ImageID', how='inner').values

    # encode ascii each string
    ascii_imgs = np.array([[j.encode("ascii", "ignore") for j in i] for i in new_refs])

    # reopen file
    with h5py.File(datapath, 'r+') as f:
        # delete the original file
        del f['images']
        # recreate with new dataset contain ID and Path
        f.create_dataset('images', ascii_imgs.shape,
                         data=ascii_imgs,
                         dtype=h5py.special_dtype(vlen=str))


if __name__ == '__main__':
    main()
