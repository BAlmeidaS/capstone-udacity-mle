import os
import h5py

import numpy as np
import pandas as pd

import ray
# from multiprocessing import cpu_count

from keras.preprocessing import image

from tqdm import tqdm

from project.utils import data_augmentation as da
import project.download_content as content


@ray.remote
def save_data(df, label_id, file_ref):
    df = df[['ImageID', 'Path']].drop_duplicates()

    with h5py.File(file_ref, 'a') as f:
        for row in df.itertuples(index=False):
            img_bin = image.load_img(row.Path, target_size=(300, 300))
            img = image.img_to_array(img_bin)

            img_normal = da.normalize(img)

            expand_dim = np.expand_dims(img_normal, axis=0)

            f.create_dataset(name=row.ImageID,
                             data=expand_dim,
                             dtype=np.float16,
                             compression='gzip',
                             compression_opts=1)


def save_batch(df, prefix):
    all_label_ids = df.LabelID.unique()

    # set file name
    folder = os.path.join(content.DATAPATH, 'MODEL', f'{prefix}')
    if not os.path.exists(folder):
        os.makedirs(folder)

    files_path = [f"{folder}/{label_id}.h5" for label_id in all_label_ids]

    futures = []

    for file_unit in files_path:
        temp_f = h5py.File(file_unit, 'w')
        temp_f.close()

    ray.init()

    try:
        for file_path, label_id in tqdm(zip(files_path, all_label_ids)):
            futures += [save_data.remote(df[df.LabelID == label_id],
                                         label_id,
                                         file_path)]

            # if len(futures) % cpu_count()*3 == 0:
            #     ray.get(futures)

        ray.get(futures)
    finally:
        ray.shutdown()


def main():
    print('creating cross_val dataset...', end='')
    cross_val_path = os.path.join(content.DATAPATH,
                                  "MODEL",
                                  "cross_val_data_preprocessed.h5")
    cross_val_metainfo = pd.read_hdf(cross_val_path, key='X', mode='r')
    save_batch(cross_val_metainfo, 'cross_val')
    print('ok!')

    print('creating test dataset...', end='')
    test_path = os.path.join(content.DATAPATH,
                             "MODEL",
                             f"test_data_preprocessed.h5")
    test_metainfo = pd.read_hdf(test_path, key='X', mode='r')
    save_batch(test_metainfo, 'test')
    print('ok!')


if __name__ == '__main__':
    main()
