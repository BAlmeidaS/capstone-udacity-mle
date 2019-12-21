import os
import h5py
import numpy as np
import ray

from itertools import zip_longest

from tqdm import tqdm

from project.utils import data_augmentation as da

import project.download_content as content

from multiprocessing import cpu_count

import uuid

# TRAIN_DATAPATH = os.path.join(content.DATAPATH, "MODEL", '39_classes_300x300.h5')
TRAIN_DATAPATH = os.path.join("/media/external", '39_classes_300x300.h5')


def save_dataset(batch_x, batch_y, file_ref, group, data_ref):
    id = uuid.uuid4()

    file_ref[group].create_dataset(name=f"{id}-x",
                                   data=batch_x,
                                   dtype=np.float16,
                                   compression='gzip',
                                   compression_opts=9)

    file_ref[group].create_dataset(name=f"{id}-y",
                                   data=batch_y,
                                   dtype=np.float16,
                                   compression='gzip',
                                   compression_opts=9)

    data_ref.append([f"{id}-x".encode("ascii", "ignore"),
                     f"{id}-y".encode("ascii", "ignore")])


def grouper(iterable, n, fillvalue=None):
    "https://docs.python.org/3/library/itertools.html#itertools-recipes"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def create_data(futures, file_ref, data_ref, group, batch_size, batch_remain):
    while len(futures):
        done_id, futures = ray.wait(futures)
        results = ray.get(done_id[0])

        batch_remain += [item for sublist in results for item in sublist]

        while len(batch_remain) > batch_size:
            topk = batch_remain[:batch_size]
            batch_remain = batch_remain[batch_size:]

            batch_x = np.concatenate([k[0] for k in topk], axis=0)
            batch_y = np.concatenate([k[1] for k in topk], axis=0)

            save_dataset(batch_x, batch_y, file_ref, group, data_ref)

    return futures, batch_remain


def main(batch_size=8, batch_images=10):
    ray.init()
    file_ref = h5py.File(TRAIN_DATAPATH, 'w')

    group_size = 300000
    batch_remain = []
    data_ref = None

    try:
        for main_group in da.get_group_imgs()[:20]:
            print("#"*20, f"processing GROUP {main_group}", "#"*20)
            images, X = da.load_data(main_group)

            futures = []

            for i, items in tqdm(enumerate(
                grouper(zip(images, X), batch_images)
            )):
                # print("iteration:", i, " -- with batch remain:", len(batch_remain))
                if i % group_size == 0:
                    print("#"*20, f"creating SUB-GROUP {main_group}", "#"*20)

                    if data_ref is None:
                        data_ref = []
                    group = str(int(i/group_size))
                    file_ref.create_group(group)

                futures += [da.batch_data_augmentation.remote(items)]
                futures += [da.batch_data_augmentation_2.remote(items)]
                futures += [da.batch_data_augmentation_3.remote(items)]
                if i % cpu_count() == 0:
                    futures, batch_remain = create_data(futures, file_ref, data_ref,
                                                        group, batch_size, batch_remain)

                if (i+1) % group_size == 0:
                    file_ref[group].create_dataset(name='data',
                                                   shape=(len(data_ref), 2),
                                                   data=data_ref,
                                                   dtype=h5py.special_dtype(vlen=str),
                                                   compression='gzip',
                                                   compression_opts=9)
                    data_ref = []

            if len(data_ref) > 0:
                file_ref[group].create_dataset(name='data',
                                               shape=(len(data_ref), 2),
                                               data=data_ref,
                                               dtype=h5py.special_dtype(vlen=str),
                                               compression='gzip',
                                               compression_opts=9)
                data_ref = []

    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()