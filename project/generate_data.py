import os
import h5py
import numpy as np
import ray

from itertools import zip_longest, cycle

from tqdm import tqdm

from project.utils import data_augmentation as da

from multiprocessing import cpu_count

import uuid

# TRAIN_DATAPATH = os.path.join(content.DATAPATH, "MODEL", '39_classes_300x300.h5')
TRAIN_DATAPATH = os.path.join("/media/external", '39_classes_300x300')


def save_dataset(x, y, file_ref):
    id = uuid.uuid4()

    file_ref.create_dataset(name=f"{id}-x",
                            data=x,
                            dtype=np.float16,
                            compression='gzip',
                            compression_opts=9)

    file_ref.create_dataset(name=f"{id}-y",
                            data=y,
                            dtype=np.float16,
                            compression='gzip',
                            compression_opts=9)


def grouper(iterable, n, fillvalue=None):
    "https://docs.python.org/3/library/itertools.html#itertools-recipes"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@ray.remote
def save_data(items, file_ref):
    processed = []

    if items is None or len(items) < 1:
        return

    for image_info, bboxes in items:
        processed += da.data_augmentation(image_info, bboxes)

    if len(processed) > 0:
        with h5py.File(file_ref, 'a') as f:
            batch_x = np.concatenate([x for x, _ in processed], axis=0)
            batch_y = np.concatenate([y for _, y in processed], axis=0)

            save_dataset(batch_x, batch_y, f)
            # for x, y in processed:
            #     save_dataset(x, y, f, group)


def main(batch_images=100):
    ray.init()

    files = [f"{TRAIN_DATAPATH}_{i}.h5" for i in range(cpu_count())]

    for f in files:
        temp_f = h5py.File(f, 'w')
        temp_f.close()

    try:
        for group in da.get_group_imgs():
            print("#"*20, f"processing GROUP {group}", "#"*20)
            images, X = da.load_data(group)

            futures = []

            for i, (items, f) in tqdm(enumerate(
                zip(grouper(zip(images, X), batch_images), cycle(files)), 1
            )):

                futures += [save_data.remote(items, f)]

                if i % (cpu_count()) == 0:
                    ray.get(futures)

    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
