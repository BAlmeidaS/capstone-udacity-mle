import os
import h5py
import numpy as np

import ray
from multiprocessing import cpu_count

from keras.preprocessing import image

from itertools import zip_longest, cycle
from operator import is_not
from functools import partial

from tqdm import tqdm

from project.utils import data_augmentation as da
import project.download_content as content

import uuid

# TRAIN_DATAPATH = os.path.join(content.DATAPATH, "MODEL", '39_classes_300x300.h5')
TRAIN_DATAPATH = os.path.join("/media/external", '39_classes_300x300')


def save_dataset(x, y, i, b, file_ref):
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

    file_ref.create_dataset(name=f"{id}-info",
                            shape=(len(i), 2),
                            data=i,
                            dtype=h5py.special_dtype(vlen=str),
                            compression='gzip',
                            compression_opts=9)

    group = f"{id}-bboxes"
    file_ref.create_group(group)

    for bboxes, info in zip(b, i):
        file_ref[group].create_dataset(name=info[0].decode('ascii'),
                                       data=bboxes,
                                       dtype=np.float16,
                                       compression='gzip',
                                       compression_opts=9)


def grouper(iterable, n, fillvalue=None):
    "https://docs.python.org/3/library/itertools.html#itertools-recipes"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_data(path):
    with h5py.File(path, 'r') as f:
        images = f['images'][:]
        X = np.array([f[i[0]][:] for i in images])

    return images, X


@ray.remote
def save_data(items, file_ref):
    # removing Nones added by grouper
    items = list(filter(partial(is_not, None), items))

    processed_b = []
    processed_x = []
    processed_y = []
    processed_i = []

    # process each image to find their targets
    for image_info, bboxes in items:
        # the '3:' removes a useless two points saved
        img_bin = image.load_img('project/' + image_info[1][3:], target_size=(300, 300))
        img = image.img_to_array(img_bin)

        # create target value
        target = da.match_bbox(bboxes, unmatch_return=True)

        infos = [image_info[0].encode("ascii", "ignore"),
                 image_info[1][3:].encode("ascii", "ignore")]

        # save the set (image, target) in processed list
        processed_b += [bboxes]
        processed_x += [np.expand_dims(img, axis=0)]
        processed_y += [np.expand_dims(target, axis=0)]
        processed_i += [np.expand_dims(infos, axis=0)]

    with h5py.File(file_ref, 'a') as f:
        # split processed list in X and  y
        batch_b = processed_b
        batch_x = np.concatenate(processed_x)
        batch_y = np.concatenate(processed_y)
        batch_i = np.concatenate(processed_i)

        # save these batches
        save_dataset(batch_x, batch_y, batch_i, batch_b, f)


def save_batch(images, X, prefix, batch_images=300):
    # set file name
    file_ref = os.path.join(content.DATAPATH, 'MODEL', f'39_classes_300x300_{prefix}')
    files = [f"{file_ref}_{i}.h5" for i in range(cpu_count())]

    futures = []

    for file_unit in files:
        temp_f = h5py.File(file_unit, 'w')
        temp_f.close()

    ray.init()

    try:
        # iterate in batches inside all images and X's
        for i, (items, file_unit) in tqdm(enumerate(
            zip(grouper(zip(images, X), batch_images), cycle(files)), 1
        )):
            futures += [save_data.remote(items, file_unit)]

            if i % (cpu_count()) == 0:
                ray.get(futures)

        ray.get(futures)
    finally:
        ray.shutdown()


def main():
    print('creating cross_val dataset...', end='')
    valpath = os.path.join(content.DATAPATH, 'MODEL', 'cross_val_data_300_vgg.h5')
    images, X = load_data(valpath)
    save_batch(images, X, 'cross_val')
    print('ok!')

    print('creating test dataset...', end='')
    testpath = os.path.join(content.DATAPATH, 'MODEL', 'test_data_300_vgg.h5')
    images, X = load_data(testpath)
    save_batch(images, X, 'test')
    print('ok!')


if __name__ == '__main__':
    main()
